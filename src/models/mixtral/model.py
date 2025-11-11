import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import repeat
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

class MOE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ffn_hidden_dim: int,
                 n_experts: int,
                 n_experts_per_tok: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [FeedForwardBlock(input_dim, ffn_hidden_dim) for _ in range(n_experts)]
        )
        self.gate = nn.Linear(input_dim, n_experts, bias=False)
        self.n_experts_per_tok = n_experts_per_tok
    
    def forward(self, x):
        # x : (batch, seqlen, embed_dim)
        logits = self.gate(x) # (batch, seqlen, n_experts)

        (weights, selected_experts) = torch.topk(logits, self.n_experts_per_tok, dim=-1) # (batch, seqlen, n_experts_per_tok)
        weights = F.softmax(weights, dim=-1)

        x_repeat = repeat(x, "b l d -> b l exps d", exps=self.n_experts_per_tok)
        output = torch.empty_like(x_repeat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i)
            output[mask] = expert(x_repeat[mask]) * weights[mask][None, :]
        
        output = output.sum(dim=2)
        return output

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 ffn_hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, ffn_hidden_dim)
        self.linear2 = nn.Linear(ffn_hidden_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, ffn_hidden_dim)

    def forward(self, x):
        x = F.silu(self.linear1(x)) * self.linear3(x)
        x = self.linear2(x)
        return x