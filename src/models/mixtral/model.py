import torch
from torch import nn, Tensor


from src.layers.attention import GroupedQueryRotaryAttention
from src.layers.normalization import RMSNorm


class Mixtral(nn.Module):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 n_experts: int,
                 n_experts_per_tok: int,
                 vocab_size: int,
                 block_size: int,
                 n_kv_heads: int,
                 ffn_hidden_dim: int,
                 rotary_base: int=1e5,
                 norm_eps: float=1e-5,
                 use_flash: bool=True):
        super().__init__()
        if n_kv_heads is None:
            n_kv_heads = n_heads

        if ffn_hidden_dim is None:
            ffn_hidden_dim = int(4 * embed_dim * (2/3))

            if ffn_hidden_dim % 256 != 0:
                ffn_hidden_dim += 256 - ffn_hidden_dim % 256

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.rotary_base = rotary_base
        self.ffn_hidden_dim = ffn_hidden_dim
        self.norm_eps = norm_eps

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, ffn_hidden_dim, n_heads, n_kv_heads, n_experts,
                         n_experts_per_tok, block_size, rotary_base, norm_eps, use_flash)
        ])
        pass
class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 ffn_hidden_dim: int,
                 n_heads:int,
                 n_kv_heads: int,
                 n_experts: int,
                 n_experts_per_tok: int,
                 block_size: int,
                 rotary_base: int,
                 norm_eps: float,
                 use_flash: bool):
        super().__init__()

        self.norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.attn = GroupedQueryRotaryAttention(embed_dim, n_heads, n_kv_heads, dropout_p=0.,
                                                apply_rotary_embedding=True, rotary_base=rotary_base,
                                                max_seqlen=block_size, bias=False, use_flash=use_flash)
        self.norm2 = RMSNorm(embed_dim, eps=norm_eps)
        self.moe = MOE(embed_dim, ffn_hidden_dim, n_experts, n_experts_per_tok)

        pass