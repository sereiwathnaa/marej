"""Standard multi-head self-attention with KV cache (GPT-2 style)."""

from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from .normalization import LayerNorm
from .dotproductattention import DotProductAttention


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


if __name__ == "__main__":
    # Self-checks: flash vs manual path, causality, and cached incremental decoding.
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, l, embed_dim, n_heads = 2, 100, 64, 4
    layer = MultiheadAttention(embed_dim, n_heads, embed_dim // n_heads, dropout_p=0.).to(device).eval()
    x = torch.randn(b, l, embed_dim, device=device)
    causal_mask = lambda n: torch.ones((n, n), device=device).triu(1)
    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()

    with torch.no_grad():
        full = layer(x, causal_mask(l))
        layer.attn.use_flash = False
        manual = layer(x, causal_mask(l))
        layer.attn.use_flash = True
        print(f"flash vs manual softmax | rel err {rel(full, manual):.2e}")

        x2 = x.clone()
        x2[:, l // 2:] = torch.randn_like(x2[:, l // 2:])
        print(f"causal leak | {(full[:, :l // 2] - layer(x2, causal_mask(l))[:, :l // 2]).abs().max():.1e}")

        layer.clear_kv_cache()
        prefill = 37
        parts = [layer(x[:, :prefill], causal_mask(prefill), use_kv_cache=True)]
        for t in range(prefill, l):
            parts.append(layer(x[:, t:t + 1], causal_mask(1), use_kv_cache=True))
        incremental = torch.cat(parts, dim=1)
    print(f"full vs incremental (cached) | max err {(full - incremental).abs().max():.2e} "
          f"| cache seqlen {layer.get_kv_cache_seqlen()} (expected {l})")
    print(layer)
