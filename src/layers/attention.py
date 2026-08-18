import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from .normalization import LayerNorm, RMSNorm
from typing import Tuple


class DotProductAttention(nn.Module):
    """Scaled dot-product attention over (batch, heads, seqlen, dim_head) tensors.

    attn_mask: (query seqlen, key seqlen); nonzero/True = NOT allowed to attend.
    Returns (batch, seqlen, heads * dim_head).
    """

    def __init__(self, use_flash: bool=True, dropout_p: float=0.):
        super().__init__()
        self.dropout = dropout_p
        self.use_flash = use_flash

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor=None):
        dropout_p = self.dropout if self.training else 0.
        if self.use_flash:
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.bool().logical_not(), dropout_p=dropout_p)
            else:
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=False)
        else:
            logits = torch.einsum("b h i d, b h j d -> b h i j", query, key) * query.shape[-1] ** -0.5
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.bool(), value=-1e9)
            attn = F.softmax(logits, dim=-1)
            attn = F.dropout(attn, self.dropout, training=self.training)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, value)

        return rearrange(out, "b h l d -> b l (h d)")


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


class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 dropout_p: float=0.,
                 bias: bool=True,
                 use_flash=True,
                 batch_first: bool=True):
        super().__init__()
        assert embed_dim % n_heads == 0
        inner_dim = dim_head * n_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        # Unused in forward; kept so existing checkpoints (which contain attn.norm.*) still load.
        self.norm = LayerNorm(embed_dim)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)

        self.kv_cache: Tuple[torch.Tensor] = None
        self.use_flash = hasattr(F, "scaled_dot_product_attention") and use_flash
        self.attn = DotProductAttention(self.use_flash, dropout_p=dropout_p)

    def forward(self,
                x: Tensor,
                attn_mask: Tensor=None,
                use_kv_cache: bool=False):

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.n_heads), qkv)

        if use_kv_cache:
            if self.training:
                raise RuntimeError("MultiheadAttention must be in .eval() mode if using KV caching.")
            if self.kv_cache is not None:
                (key_cache, value_cache) = self.kv_cache
                key = torch.cat([key_cache, key], dim=2)
                value = torch.cat([value_cache, value], dim=2)

                if attn_mask is not None:
                    cache_attn_mask = torch.zeros((len(attn_mask), self.get_kv_cache_seqlen()), dtype=attn_mask.dtype, device=attn_mask.device)
                    attn_mask = torch.cat([cache_attn_mask, attn_mask], dim=1)

            self.kv_cache = (key.detach(), value.detach())

        attn_output = self.attn(query, key, value, attn_mask)
        out = self.to_out(attn_output)
        return out

    def get_kv_cache_seqlen(self):
        if self.kv_cache is None:
            return 0
        else:
            return self.kv_cache[0].shape[2]

    def clear_kv_cache(self):
        self.kv_cache = None

    def __repr__(self):
        return (f'MultiheadAttention(embed_dim={self.embed_dim}, '
                f'n_heads={self.n_heads})')


class FixedSparseAttention(nn.Module):
    """Block-local causal attention: each block of `block_size` tokens attends only within itself.

    Note: heads are not split; attention runs single-headed over inner_dim = dim_head * n_heads.
    seqlen must be divisible by block_size.
    """

    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 block_size: int=8,
                 use_flash: bool=True):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.block_size = block_size
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, embed_dim)
        self.use_flash = hasattr(F, "scaled_dot_product_attention") and use_flash

    def forward(self, x: Tensor):
        b, seqlen, _ = x.shape
        num_blocks = seqlen // self.block_size

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = map(lambda t: rearrange(t, "b (nb bs) d -> b nb bs d", nb=num_blocks, bs=self.block_size), qkv)
        attn_mask = torch.ones((self.block_size, self.block_size), device=x.device, dtype=torch.bool).triu(1)
        if self.use_flash:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.logical_not(), is_causal=False)
        else:
            logits = torch.einsum("b n i d, b n j d -> b n i j", query, key) * query.shape[-1] ** -0.5
            logits.masked_fill_(attn_mask, value=-1e9)
            attn = F.softmax(logits, dim=-1)
            out = torch.einsum("b n i j, b n j d -> b n i d", attn, value)

        out = rearrange(out, "b nb bs d -> b (nb bs) d")
        out = self.to_out(out)
        return out


class StridedSparseAttention(nn.Module):
    """Block-local causal attention plus full attention to the previous block.

    Note: heads are not split; attention runs single-headed over inner_dim = dim_head * n_heads.
    seqlen must be divisible by block_size.
    """

    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 block_size: int=8):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.block_size = block_size
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, embed_dim)

    def forward(self,
                x: Tensor):
        b, seqlen, _ = x.shape
        num_blocks = seqlen // self.block_size
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b (nb bs) d -> b nb bs d", nb=num_blocks, bs=self.block_size), qkv)
        q_prev = q[:, 1:]
        k_prev = k[:, :-1]
        v_prev = v[:, :-1]

        # attend to current subblock
        logits = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
        causal_attn_mask = torch.ones((self.block_size, self.block_size), device=x.device, dtype=torch.bool).triu(1)
        logits = torch.masked_fill(logits, mask=causal_attn_mask, value=-1e9)

        # attend to previous subblock
        # (batch, num_blocks - 1, block_size, block_size)
        logits_prev = q_prev @ k_prev.transpose(-1, -2) * q.shape[-1] ** -0.5

        prev_attn_mask = torch.ones((self.block_size, self.block_size), device=x.device, dtype=torch.bool).tril(0)
        logits_prev = torch.masked_fill(logits_prev, mask=prev_attn_mask, value=-1e9)

        # first block has no previous block
        padding_neginf = torch.full_like(logits[:, :1], -1e9)
        # (batch, num_blocks, block_size, block_size)
        logits_prev = torch.cat([padding_neginf, logits_prev], dim=1)

        # (batch, num_blocks, block_size, block_size * 2)
        logits_prev_and_curr = torch.cat([logits_prev, logits], dim=-1)
        attn = F.softmax(logits_prev_and_curr, dim=-1)
        (attn_prev, attn) = attn.chunk(2, dim=-1)
        attn_output = attn @ v
        attn_output_prev = attn_prev[:, 1:] @ v_prev

        attn_output[:, 1:] += attn_output_prev
        attn_output = rearrange(attn_output, "b nb bs d -> b (nb bs) d")
        return self.to_out(attn_output)
