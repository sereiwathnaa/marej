import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from .normalization import LayerNorm

class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 dropout_p: float=0.,
                 use_flash: bool=True,
                 bias: bool=True,
                 batch_first: bool=True):
        super().__init__()
        assert embed_dim % n_heads == 0
        inner_dim = dim_head * n_heads
        self.n_heads = n_heads
        self.norm = LayerNorm(embed_dim)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)       

        self.kv_cache: tuple[torch.Tensor]
        self.use_flash = hasattr(F, "scaled_dot_product_attention")
        self.attn = DotProductAttention(use_flash=use_flash, dropout_p=dropout_p)
    
    # def forward(self, x):



class DotProductAttention(nn.Module):
    def __int__(self, use_flash: bool=True, dropout_p: float=0.):
        super().__init__()
        self.dropout = dropout_p
        self.use_flash = use_flash
    
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor=None):
        if self.use_flash:
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.bool().logical_not(), dropout_p=self.dropout if self.training else 0.)
            else:
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0., is_causal=True)
            out = rearrange(out, "b h l d -> b l (h d)")
        else:
            logits = torch.einsum("b h i d, b h j d -> b h i j", query, key) * key.shape[-1] ** -0.5
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.bool(), value=float("-inf"))
            attn = F.softmax(logits, dim=-1)
            attn = F.dropout(self.dropout)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, value)
            out = rearrange(out, "b h l d -> b l (h d)")

        return out