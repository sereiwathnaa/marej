import torch
from torch import nn, Tensor
import torch.nn.functional as F

from transformers import AutoModelForCausalLM

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 ffn_hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, ffn_hidden_dim, bias=False)
        self.linear2 = nn.Linear(ffn_hidden_dim, input_dim, bias=False)
        self.linear3 = nn.Linear(input_dim, ffn_hidden_dim, bias=False)

    def forward(self, x):
        x = F.silu(self.linear1(x)) * self.linear3(x) # SwiGLU "activation"
        x = self.linear2(x)
        return x

class Llama(nn.Module):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 vocab_size: int,
                 block_size: int,
                 n_kv_heads: int=None,
                 ffn_hidden_dim: int=None,
                 rotary_base: int=10000,
                 norm_eps: float=1e-5):
        super().__init__()

        # MultiHeadRotaryAttention in this case
        if n_kv_heads is None:
            n_kv_heads = n_heads

        if ffn_hidden_dim is None:
            # Llama uses SwiGLU, ffn_hidden_dim is set to 4 * embed_dim * 2/3
            ffn_hidden_dim = int(4 * embed_dim * (2/3))

            if ffn_hidden_dim % 256 != 0:
                ffn_hidden_dim = ffn_hidden_dim + 256 - ffn_hidden_dim % 256

        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.rotary_base = rotary_base
        self.ffn_hidden_dim = ffn_hidden_dim
        self.norm_eps = norm_eps


        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, ffn_hidden_dim, n_heads, n_kv_heads, block_size, rotary_base, norm_eps, use_flash)]
        )
        pass