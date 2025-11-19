#%%
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from normalization import LayerNorm
from typing import Tuple
#%%
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
class GroupedQueryRotaryAttention(nn.Module):
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
                 batch_first: bool=True):
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
    
    def forward(self,
                 x: Tensor,
                 attn_mask: Tensor=None,
                 use_kv_cache: bool=False,
                 rotation_matr: tuple[Tensor, Tensor]=None):

        q = self.to_q(x)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.n_heads)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "b l (h d) -> b h l d", h=self.n_kv_heads), kv)

        if self.apply_rotary_embedding:
            offset = self.get_kv_cache_seqlen() if use_kv_cache else 0
            seqlen = q.shape[2]
            
            if rotation_matr is None:
                # Need enough rotation positions for offset + current sequence
                max_pos = max(offset + seqlen, self.max_seqlen)
                rotation_matr = self._compute_rotation_matrix(max_pos, q.device)

            q = self.apply_rotation_matrix(q, rotation_matr, offset)
            k = self.apply_rotation_matrix(k, rotation_matr, offset )

        if use_kv_cache:
            if self.training:
                raise RuntimeError("GroupQueryRotaryAttention must be in .eval() mode if using KV caching")
            if self.kv_cache is not None:
                k_cache, v_cache = self.kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)

            if attn_mask is not None:
                cache_attn_mask = torch.zeros((len(attn_mask), self.get_kv_cache_seqlen()), device=attn_mask.device, dtype=attn_mask.dtype)
                attn_mask = torch.cat([cache_attn_mask, attn_mask], dim=1)

            self.kv_cache = (k.detach(), v.detach())

        if self.n_heads != self.n_kv_heads:
            k = repeat(k, "b kv_h l d -> b (kv_h rep) l d", rep=self.n_heads//self.n_kv_heads)
            v = repeat(v, 'b kv_h l d -> b (kv_h rep) l d', rep=self.n_heads//self.n_kv_heads)
        
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
        angle = torch.outer(
            torch.arange(seqlen, device=device),
            (1. / self.rotary_base ** (2 * torch.arange(self.dim_head // 2, device=device) / self.dim_head))
        )
        cos_A = torch.stack([angle.cos(), angle.cos()], dim=2)
        sin_A = torch.stack([-angle.sin(), angle.sin()], dim=2)
        rotation_matr = (cos_A, sin_A)
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
        qk = rearrange(qk, "... (d j) -> ... d j", j=2)
        qk_rotated = (cos_A[offset:offset+seqlen] * qk
                      + sin_A[offset:offset+seqlen] * torch.flip(qk, dims=[-1]))
        qk_rotated = rearrange(qk_rotated, "... d j -> ... (d j)")
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

#%%
import torch
import torch.nn as nn

# Example configuration for GroupedQueryRotaryAttention
embed_dim = 1024
n_heads = 16
n_kv_heads = 4  # Grouped query attention with fewer KV heads
dropout_p = 0.1
apply_rotary_embedding = True
rotary_base = 10000
max_seqlen = 1024
bias = True
use_flash = True
batch_first = True

# Instantiate the attention module
attention = GroupedQueryRotaryAttention(
    embed_dim=embed_dim,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    dropout_p=dropout_p,
    apply_rotary_embedding=apply_rotary_embedding,
    rotary_base=rotary_base,
    max_seqlen=max_seqlen,
    bias=bias,
    use_flash=use_flash,
    batch_first=batch_first
)

# Prepare input tensor (batch_size, seq_len, embed_dim)
# batch_size = 2
# seq_len = 64
# x = torch.randn(batch_size, seq_len, embed_dim)

# # Set to eval mode for KV cache usage
# attention.eval()

# # Forward pass without KV cache
# output_no_cache = attention(x)

# print(f"Output shape without KV cache: {output_no_cache.shape}")

# # Forward pass with KV cache (simulate incremental generation)
# attention.clear_kv_cache()  # Ensure cache is cleared

# # First chunk
# x_chunk1 = x[:, :32, :]  # First 32 tokens
# output_chunk1 = attention(x_chunk1, use_kv_cache=True)

# print(f"Output shape for chunk 1: {output_chunk1.shape}")
# print(f"KV cache sequence length after chunk 1: {attention.get_kv_cache_seqlen()}")

# # Second chunk (continuing from cache)
# x_chunk2 = x[:, 32:64, :]  # Next 32 tokens
# output_chunk2 = attention(x_chunk2, use_kv_cache=True)

# print(f"Output shape for chunk 2: {output_chunk2.shape}")
# print(f"KV cache sequence length after chunk 2: {attention.get_kv_cache_seqlen()}")

# # Clear cache when done
# attention.clear_kv_cache()

#%%

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

        self.kv_cache: Tuple[torch.Tensor] = None
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

#%%
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

# %%
class FixedSparseAttention(nn.Module):
    def __init__(self,
                 embed_dim: int=512,
                 n_heads: int=8,
                 dim_head: int=64,
                 block_size: int=8,
                 use_flash: bool=True):
        super().__init__()
        # assert embed_dim % n_heads == 0
        inner_dim = dim_head * n_heads
        self.block_size = block_size
        self.norm = LayerNorm(embed_dim)
        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3)
        self.to_out = nn.Linear(inner_dim, embed_dim)
        self.use_flash = hasattr(F, "scaled_dot_product_attention") and use_flash

    def forward(self, x: Tensor):
        b, seqlen, _ = x.shape
        
        # Handle padding if sequence length is not divisible by block size

            
        padded_seqlen = x.shape[1]
        num_blocks = padded_seqlen // self.block_size
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # Rearrange to (batch, num_blocks, block_size, dim)
        query, key, value = map(lambda t: rearrange(t, "b (nb bs) d -> b nb bs d", nb=num_blocks, bs=self.block_size), qkv)

        # Causal mask for within-block attention
        # 1 (True) means mask out (future positions)
        attn_mask = torch.ones((self.block_size, self.block_size), device=x.device, dtype=torch.bool).triu(1)

        if self.use_flash:
            # scaled_dot_product_attention expects (batch, heads, seqlen, dim)
            # We treat blocks as heads for parallel computation
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.logical_not(), is_causal=False)
            print(out)
        else:
            scale = query.shape[-1] ** -0.5
            # Use single-letter subscripts for einsum: b=batch, n=num_blocks, i/j=block_size, d=dim
            logits = torch.einsum("b n i d, b n j d -> b n i j", query, key) * scale
            logits.masked_fill_(attn_mask, value=-1e9)
            attn = F.softmax(logits, dim=-1)
            print(logits)
            out = torch.einsum("b n i j, b n j d -> b n i d", attn, value)
        
        out = rearrange(out, "b nb bs d -> b (nb bs) d")
        

            
        out = self.to_out(out)
        return out
# %%
