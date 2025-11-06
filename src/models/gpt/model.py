import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.normalization import LayerNorm
from layers.attention import MultiheadAttention


class GPT(nn.Module):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 vocab_size: int,
                 block_size: int,
                 dropout_p: float):
        super().__init__()
        self.dim_head = embed_dim // n_heads
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.dropout = nn.Dropout(dropout_p)

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(block_size, embed_dim)

        # self.decoder_blocks = nn.ModuleList(
            # [DecoderBlock(embed_dim, n_heads, dropout_p) for _ in range(n_layers)
        # )
        pass

class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 dim_head: int,
                 dropout_p: float,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, n_heads, dim_head, dropout_p, use_flash=True)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim, embed_dim * 4)
    pass

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x