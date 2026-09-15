"""Block-sparse attention patterns (Sparse Transformer style), single-headed over inner_dim.

Both layers share the interface of the other attention layers: forward(x, attn_mask=None, use_kv_cache=False),
`kv_cache` of shape (batch, 1, cache_seqlen, inner_dim), get_kv_cache_seqlen() and clear_kv_cache().

Two execution paths give identical results:
* blocked (training / full sequences whose length is a multiple of block_size): the original block-wise code.
* masked  (KV-cached decoding, or lengths not divisible by block_size): dense attention of the new queries
  against only the cached keys the pattern can reach, under the explicit allowed-positions mask.
"""

from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange


class _BlockSparseAttention(nn.Module):
    """Shared projections, KV cache and dispatch; subclasses define the sparsity pattern."""

    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 dim_head: int,
                 block_size: int,
                 use_flash: bool):
        super().__init__()
        inner_dim = dim_head * n_heads
        self.block_size = block_size
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, embed_dim)
        self.use_flash = hasattr(F, "scaled_dot_product_attention") and use_flash
        self.kv_cache: Tuple[Tensor, Tensor] = None

    # ---- pattern definition (absolute positions) ----
    def _allowed(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        """Bool mask (len(q_pos), len(k_pos)): True where query at q_pos may attend to key at k_pos."""
        raise NotImplementedError

    def _first_key(self, offset: int) -> int:
        """Earliest key position any query at position >= offset can attend to."""
        raise NotImplementedError

    def _blocked(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Block-wise implementation for (b, seqlen, inner_dim) inputs with seqlen % block_size == 0."""
        raise NotImplementedError

    def forward(self,
                x: Tensor,
                attn_mask: Tensor=None,
                use_kv_cache: bool=False):
        if attn_mask is not None and attn_mask.bool().tril().any():
            raise ValueError("Sparse attention is always causal; attn_mask may only mask future positions.")

        seqlen = x.shape[1]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        offset = 0
        if use_kv_cache:
            if self.training:
                raise RuntimeError(f"{type(self).__name__} must be in .eval() mode if using KV caching.")
            offset = self.get_kv_cache_seqlen()
            k_all, v_all = k.unsqueeze(1), v.unsqueeze(1)                # (b, 1, l, inner_dim)
            if self.kv_cache is not None:
                k_cache, v_cache = self.kv_cache
                k_all = torch.cat([k_cache, k_all], dim=2)
                v_all = torch.cat([v_cache, v_all], dim=2)
            self.kv_cache = (k_all.detach(), v_all.detach())
            k, v = k_all[:, 0], v_all[:, 0]

        if not use_kv_cache and seqlen % self.block_size == 0:
            out = self._blocked(q, k, v)
        else:
            start = self._first_key(offset)
            q_pos = torch.arange(offset, offset + seqlen, device=x.device)
            k_pos = torch.arange(start, offset + seqlen, device=x.device)
            allowed = self._allowed(q_pos[:, None], k_pos[None, :])
            out = self._masked(q, k[:, start:], v[:, start:], allowed)
        return self.to_out(out)

    def _masked(self, q: Tensor, k: Tensor, v: Tensor, allowed: Tensor) -> Tensor:
        """Dense single-head attention (b, lq, d) x (b, lk, d) under a bool allowed mask (lq, lk)."""
        if self.use_flash:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, is_causal=False)
        logits = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
        logits = logits.masked_fill(~allowed, value=-1e9)
        return F.softmax(logits, dim=-1) @ v

    def get_kv_cache_seqlen(self):
        if self.kv_cache is None:
            return 0
        else:
            return self.kv_cache[0].shape[2]

    def clear_kv_cache(self):
        self.kv_cache = None

    def __repr__(self):
        return (f'{type(self).__name__}(embed_dim={self.to_qkv.in_features}, '
                f'inner_dim={self.to_out.in_features}, block_size={self.block_size})')


class FixedSparseAttention(_BlockSparseAttention):
    """Block-local causal attention: each block of `block_size` tokens attends only within itself.

    Note: heads are not split; attention runs single-headed over inner_dim = dim_head * n_heads.
    """

    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 block_size: int=8,
                 use_flash: bool=True):
        super().__init__(embed_dim, n_heads, dim_head, block_size, use_flash)

    def _allowed(self, q_pos, k_pos):
        return (k_pos <= q_pos) & (k_pos // self.block_size == q_pos // self.block_size)

    def _first_key(self, offset):
        return (offset // self.block_size) * self.block_size          # start of the current block

    def _blocked(self, q, k, v):
        num_blocks = q.shape[1] // self.block_size
        query, key, value = map(lambda t: rearrange(t, "b (nb bs) d -> b nb bs d", nb=num_blocks, bs=self.block_size), (q, k, v))
        attn_mask = torch.ones((self.block_size, self.block_size), device=q.device, dtype=torch.bool).triu(1)
        if self.use_flash:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.logical_not(), is_causal=False)
        else:
            logits = torch.einsum("b n i d, b n j d -> b n i j", query, key) * query.shape[-1] ** -0.5
            logits.masked_fill_(attn_mask, value=-1e9)
            attn = F.softmax(logits, dim=-1)
            out = torch.einsum("b n i j, b n j d -> b n i d", attn, value)
        return rearrange(out, "b nb bs d -> b (nb bs) d")


class StridedSparseAttention(_BlockSparseAttention):
    """Block-local causal attention plus the tail of the previous block.

    Query at local position r in a block attends to positions 0..r of its own block and to positions
    r+1..block_size-1 of the previous block, i.e. a causal sliding window of exactly `block_size` tokens.

    Note: heads are not split; attention runs single-headed over inner_dim = dim_head * n_heads.
    """

    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 block_size: int=8,
                 use_flash: bool=True):
        super().__init__(embed_dim, n_heads, dim_head, block_size, use_flash)

    def _allowed(self, q_pos, k_pos):
        return (k_pos <= q_pos) & (k_pos > q_pos - self.block_size)

    def _first_key(self, offset):
        return max(0, offset - self.block_size + 1)                  # window of the first new query

    def _blocked(self, q, k, v):
        num_blocks = q.shape[1] // self.block_size
        q, k, v = map(lambda t: rearrange(t, "b (nb bs) d -> b nb bs d", nb=num_blocks, bs=self.block_size), (q, k, v))
        q_prev = q[:, 1:]
        k_prev = k[:, :-1]
        v_prev = v[:, :-1]

        # attend to current subblock
        logits = q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5
        causal_attn_mask = torch.ones((self.block_size, self.block_size), device=q.device, dtype=torch.bool).triu(1)
        logits = torch.masked_fill(logits, mask=causal_attn_mask, value=-1e9)

        # attend to previous subblock
        # (batch, num_blocks - 1, block_size, block_size)
        logits_prev = q_prev @ k_prev.transpose(-1, -2) * q.shape[-1] ** -0.5

        prev_attn_mask = torch.ones((self.block_size, self.block_size), device=q.device, dtype=torch.bool).tril(0)
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
        return rearrange(attn_output, "b nb bs d -> b (nb bs) d")


if __name__ == "__main__":
    # Self-checks: each sparse pattern must equal dense single-head attention under the explicit
    # allowed-positions mask (blocked path and non-divisible-length masked path), be causal, and give
    # the same outputs with cached incremental decoding.
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, l, embed_dim, n_heads, dim_head, bs = 2, 96, 64, 4, 16, 16
    x = torch.randn(b, l, embed_dim, device=device)
    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()

    def dense_reference(layer, x, allowed):
        """Dense single-head attention over inner_dim using the layer's own weights and an allowed mask."""
        q, k, v = layer.to_qkv(x).chunk(3, dim=-1)
        logits = (q @ k.transpose(-1, -2) * q.shape[-1] ** -0.5).masked_fill(~allowed, float("-inf"))
        return layer.to_out(logits.softmax(-1) @ v)

    def positions(n):
        return torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")

    with torch.no_grad():
        for name, layer in (
                ("FixedSparse (flash)", FixedSparseAttention(embed_dim, n_heads, dim_head, bs, use_flash=True)),
                ("FixedSparse (manual)", FixedSparseAttention(embed_dim, n_heads, dim_head, bs, use_flash=False)),
                ("StridedSparse (flash)", StridedSparseAttention(embed_dim, n_heads, dim_head, bs, use_flash=True)),
                ("StridedSparse (manual)", StridedSparseAttention(embed_dim, n_heads, dim_head, bs, use_flash=False))):
            layer = layer.to(device).eval()
            out = layer(x)
            i, j = positions(l)
            ref_err = rel(dense_reference(layer, x, layer._allowed(i, j)), out)

            odd = l - 5   # length not divisible by block_size -> masked path
            i, j = positions(odd)
            odd_err = rel(dense_reference(layer, x[:, :odd], layer._allowed(i, j)), layer(x[:, :odd]))

            x2 = x.clone()
            x2[:, l // 2:] = torch.randn_like(x2[:, l // 2:])
            leak = (out[:, :l // 2] - layer(x2)[:, :l // 2]).abs().max().item()

            layer.clear_kv_cache()
            prefill = 37
            parts = [layer(x[:, :prefill], use_kv_cache=True)]
            for t in range(prefill, l):
                parts.append(layer(x[:, t:t + 1], use_kv_cache=True))
            cache_err = (out - torch.cat(parts, dim=1)).abs().max().item()
            cache_len = layer.get_kv_cache_seqlen()
            layer.clear_kv_cache()

            print(f"{name:24s} | vs dense reference rel err {ref_err:.1e} (len {l}) {odd_err:.1e} (len {odd}) "
                  f"| causal leak {leak:.1e} | full vs cached max err {cache_err:.1e} | cache seqlen {cache_len} (expected {l})")
