"""Grouped-query attention with rotary embedding (Llama / Qwen style) and KV cache."""

from typing import Tuple

import torch
from torch import nn, Tensor
from einops import rearrange, repeat

from .normalization import RMSNorm
from .dotproductattention import DotProductAttention


class GroupedQueryRotaryAttention(nn.Module):
    """Grouped-query attention with optional rotary embedding (rotate-half convention) and KV cache.

    MultiheadAttention == GroupedQueryRotaryAttention(n_kv_heads=n_heads, apply_rotary_embedding=False).
    """

    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 dropout_p: float,
                 apply_rotary_embedding: bool,
                 rotary_base: int=10000,
                 max_seqlen: int=4096,
                 bias: bool=True,
                 use_flash: bool=True,
                 batch_first: bool=True,
                 qk_norm: bool=False):
        super().__init__()
        assert embed_dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.apply_rotary_embedding = apply_rotary_embedding
        self.rotary_base = rotary_base
        self.max_seqlen = max_seqlen
        self.dim_head = embed_dim // n_heads
        self.kv_cache: Tuple[Tensor, Tensor] = None

        self.attention = DotProductAttention(use_flash=use_flash, dropout_p=dropout_p)
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_kv = nn.Linear(embed_dim, n_kv_heads * self.dim_head * 2, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Per-head RMSNorm on q and k before rotary embedding (Qwen3-style)
        self.q_norm = RMSNorm(self.dim_head) if qk_norm else None
        self.k_norm = RMSNorm(self.dim_head) if qk_norm else None

    def forward(self,
                 x: Tensor,
                 attn_mask: Tensor=None,
                 use_kv_cache: bool=False,
                 rotation_matr: tuple[Tensor, Tensor]=None):

        q = self.to_q(x)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.n_kv_heads), kv)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.apply_rotary_embedding:
            offset = self.get_kv_cache_seqlen() if use_kv_cache else 0
            seqlen = q.shape[2]

            if rotation_matr is None:
                # Need enough rotation positions for offset + current sequence
                max_pos = max(offset + seqlen, self.max_seqlen)
                rotation_matr = self._compute_rotation_matrix(max_pos, q.device)

            q = self.apply_rotation_matrix(q, rotation_matr, offset)
            k = self.apply_rotation_matrix(k, rotation_matr, offset)

        if use_kv_cache:
            if self.training:
                raise RuntimeError("GroupedQueryRotaryAttention must be in .eval() mode if using KV caching")
            if self.kv_cache is not None:
                k_cache, v_cache = self.kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)

            if attn_mask is not None:
                cache_attn_mask = torch.zeros((len(attn_mask), self.get_kv_cache_seqlen()), device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([cache_attn_mask, attn_mask], dim=1)

            self.kv_cache = (k.detach(), v.detach())

        if self.n_heads != self.n_kv_heads:
            k = repeat(k, "b kv_h l d -> b (kv_h rep) l d", rep=self.n_heads // self.n_kv_heads)
            v = repeat(v, "b kv_h l d -> b (kv_h rep) l d", rep=self.n_heads // self.n_kv_heads)

        attn_output = self.attention(q, k, v, attn_mask)
        out = self.to_out(attn_output)
        return out

    def compute_rotation_matrix(self):
        """Compute rotation matrix for the max sequence length."""
        return self._compute_rotation_matrix(self.max_seqlen, None)

    def _compute_rotation_matrix(self, seqlen: int, device):
        """Internal method to compute rotation matrix for any sequence length."""
        if device is None:
            device = 'cpu'
        positions = torch.arange(seqlen, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.rotary_base ** (2 * torch.arange(self.dim_head // 2, device=device, dtype=torch.float32) / self.dim_head)
        )
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        rotation_matr = (emb.cos(), emb.sin())
        return rotation_matr

    def apply_rotation_matrix(self,
                              qk: Tensor,
                              rotation_matr: tuple[Tensor, Tensor],
                              offset: int):
        """rotate qk -> (batch, n_heads, seqlen, dim_head) by `rotation_matr`
        we define: rotation_matr: (cosA, sinA) where cosA := (cosA, cosA)
                                                     sinA := (-sinA, sinA)
        """
        (cos_A, sin_A) = rotation_matr
        seqlen = qk.shape[2]
        cos_A = cos_A[offset:offset+seqlen].to(dtype=qk.dtype, device=qk.device)
        sin_A = sin_A[offset:offset+seqlen].to(dtype=qk.dtype, device=qk.device)
        qk_half = torch.cat((-qk[..., qk.shape[-1] // 2:], qk[..., :qk.shape[-1] // 2]), dim=-1)
        qk_rotated = qk * cos_A.unsqueeze(0).unsqueeze(0) + qk_half * sin_A.unsqueeze(0).unsqueeze(0)
        return qk_rotated

    def get_kv_cache_seqlen(self):
        """Gets sequence length of kv_cache."""
        if self.kv_cache is None:
            return 0
        else:
            # kv_cache[0] and [1] are shape (batch, n_heads, cache_seqlen, per_head_dim)
            return self.kv_cache[0].shape[2]

    def clear_kv_cache(self):
        """Clears kv_cache."""
        self.kv_cache = None

    def __repr__(self):
        return (f'GroupedQueryRotaryAttention(embed_dim={self.embed_dim}, '
                f'n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, '
                f'apply_rotary_embedding={self.apply_rotary_embedding})')


if __name__ == "__main__":
    # Self-checks: rotary relative-position property, GQA == MHA with shared kv heads,
    # no-rotary == MultiheadAttention, causality, and cached incremental decoding (rotary offsets).
    from .multiheadattention import MultiheadAttention

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, l, embed_dim, n_heads = 2, 100, 64, 4
    x = torch.randn(b, l, embed_dim, device=device)
    causal_mask = lambda n: torch.ones((n, n), device=device).triu(1)
    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()

    layer = GroupedQueryRotaryAttention(embed_dim, n_heads, n_heads, 0., apply_rotary_embedding=True, max_seqlen=l).to(device).eval()

    # rotary: <rot(q, i), rot(k, j)> must depend only on i - j
    q, k = torch.randn(1, 1, 1, layer.dim_head, device=device), torch.randn(1, 1, 1, layer.dim_head, device=device)
    rot = layer._compute_rotation_matrix(64, device)
    dots = [(layer.apply_rotation_matrix(q, rot, 5 + s) * layer.apply_rotation_matrix(k, rot, 2 + s)).sum() for s in range(0, 40, 7)]
    print(f"rotary relative-position invariance | max spread {(max(dots) - min(dots)).abs():.1e}")

    with torch.no_grad():
        full = layer(x, causal_mask(l))

        # no rotary + n_kv_heads == n_heads must equal MultiheadAttention with the same weights
        gqra_plain = GroupedQueryRotaryAttention(embed_dim, n_heads, n_heads, 0., apply_rotary_embedding=False).to(device).eval()
        mha = MultiheadAttention(embed_dim, n_heads, embed_dim // n_heads, 0.).to(device).eval()
        mha.to_qkv.weight.copy_(torch.cat([gqra_plain.to_q.weight, gqra_plain.to_kv.weight]))
        mha.to_qkv.bias.copy_(torch.cat([gqra_plain.to_q.bias, gqra_plain.to_kv.bias]))
        mha.to_out.load_state_dict(gqra_plain.to_out.state_dict())
        print(f"no-rotary GQRA vs MultiheadAttention | rel err {rel(mha(x, causal_mask(l)), gqra_plain(x, causal_mask(l))):.2e}")

        # grouped kv (2 kv heads) must equal full-head attention whose kv weights are repeated per group
        gqa = GroupedQueryRotaryAttention(embed_dim, n_heads, 2, 0., apply_rotary_embedding=True, max_seqlen=l).to(device).eval()
        rep = n_heads // 2
        expand = lambda w: rearrange(repeat(rearrange(w, "(kv h d) ... -> kv h d ...", kv=2, h=2), "kv h d ... -> kv (h rep) d ...", rep=rep), "kv h d ... -> (kv h d) ...")
        layer.to_kv.weight.copy_(expand(gqa.to_kv.weight)); layer.to_kv.bias.copy_(expand(gqa.to_kv.bias))
        layer.to_q.load_state_dict(gqa.to_q.state_dict()); layer.to_out.load_state_dict(gqa.to_out.state_dict())
        print(f"GQA (2 kv heads) vs repeated-kv MHA | rel err {rel(layer(x, causal_mask(l)), gqa(x, causal_mask(l))):.2e}")

        x2 = x.clone()
        x2[:, l // 2:] = torch.randn_like(x2[:, l // 2:])
        print(f"causal leak | {(gqa(x, causal_mask(l))[:, :l // 2] - gqa(x2, causal_mask(l))[:, :l // 2]).abs().max():.1e}")

        for name, m in (("rotary", gqa), ("rotary + qk_norm", GroupedQueryRotaryAttention(embed_dim, n_heads, 2, 0., True, max_seqlen=l, qk_norm=True).to(device).eval())):
            full = m(x, causal_mask(l))
            m.clear_kv_cache()
            prefill = 37
            parts = [m(x[:, :prefill], causal_mask(prefill), use_kv_cache=True)]
            for t in range(prefill, l):
                parts.append(m(x[:, t:t + 1], causal_mask(1), use_kv_cache=True))
            print(f"{name:16s} | full vs incremental (cached) max err {(full - torch.cat(parts, dim=1)).abs().max():.2e} "
                  f"| cache seqlen {m.get_kv_cache_seqlen()} (expected {l})")
    print(gqa)
