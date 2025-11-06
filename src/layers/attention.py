#%%
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
                 bias: bool=True,
                 use_flash=True,
                 batch_first: bool=True):
        super().__init__()
        assert embed_dim % n_heads == 0
        inner_dim = dim_head * n_heads
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.norm = LayerNorm(embed_dim)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)       

        self.kv_cache: tuple[torch.Tensor] = None
        self.use_flash = hasattr(F, "scaled_dot_product_attention") & use_flash
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


class DotProductAttention(nn.Module):
    def __init__(self, use_flash: bool=True, dropout_p: float=0.):
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
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout if self.training else 0., is_causal=False)
            out = rearrange(out, "b h l d -> b l (h d)")
        else:
            logits = torch.einsum("b h i d, b h j d -> b h i j", query, key) * query.shape[-1] ** -0.5
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.bool(), value=-1e9)
            attn = F.softmax(logits, dim=-1)
            attn = F.dropout(attn, self.dropout)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, value)
            out = rearrange(out, "b h l d -> b l (h d)")

        return out

#%%
# torch.manual_seed(0)
# batch, seq_len, embed = 2, 16, 512
# x = torch.randn(batch, seq_len, embed)

# attn_flash = MultiheadAttention(embed_dim=embed, n_heads=8, dim_head=64, dropout_p=0.0, use_flash=True)
# attn_no_flash = MultiheadAttention(embed_dim=embed, n_heads=8, dim_head=64, dropout_p=0.0, use_flash=False)
# attn_no_flash.load_state_dict(attn_flash.state_dict())

# attn_flash.eval()
# attn_no_flash.eval()

# y_flash = attn_flash(x)
# y_no_flash = attn_no_flash(x)
# print("no mask diff:", (y_flash - y_no_flash).abs().max().item())

# mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
# y_flash_mask = attn_flash(x, attn_mask=mask)
# y_no_flash_mask = attn_no_flash(x, attn_mask=mask)
# print("causal mask diff:", (y_flash_mask - y_no_flash_mask).abs().max().item())
# # %%
# print((y_flash_mask.sum(), y_no_flash_mask.sum()))
# print(y_flash.sum(), y_no_flash.sum())
# # %%
