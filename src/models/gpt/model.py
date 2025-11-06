import torch
from torch import nn, Tensor
import torch.nn.functional as F
import sys
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

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, n_heads, self.dim_head, dropout_p) for _ in range(n_layers)]
        )

        self.layer_norm = LayerNorm(embed_dim)  

        self.output_projection = self.word_embeddings.weight

    def forward(self,
                indices: Tensor,
                use_kv_cache: bool=True,
                targets: Tensor=None,
                ):
        offset = self.get_kv_cache_seqlen() if use_kv_cache else 0
        position_indices = torch.arange(indices.shape[1], device=indices.device) + offset
        x = self.word_embeddings(indices) + self.position_embeddings(position_indices).unsqueeze(0)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, use_kv_cache)

        x = self.layer_norm(x)
        logits = x @ self.output_projection.T
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1)
            return logits, loss
        return logits
    

    @staticmethod
    def from_pretrained(model_name: str):
        from transformers import GPT2LMHeadModel
        model_names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        if model_name not in model_names:
            raise ValueError(f"Invalid model name, the {model_name} is not one of {model_names}")

        model_hf = GPT2LMHeadModel.from_pretrained(model_name)

        config = {
            'n_layers': len(model_hf.transformer.h),
            'n_heads': model_hf.transformer.h[0].attn.num_heads,
            'embed_dim': model_hf.transformer.h[0].attn.embed_dim,
            'vocab_size': model_hf.lm_head.out_features,
            'block_size': model_hf.transformer.wpe.num_embeddings,
            'dropout_p': model_hf.transformer.drop.p
        }

        model = GPT(**config)

        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        sd["output_projection"].copy_(sd_hf["transformer.wte.weight"])
        # sd["word_embeddings.weight"].copy_(sd_hf["transformer.wte.weight"])
        sd["position_embeddings.weight"].copy_(sd_hf["transformer.wpe.weight"])

        for i in range(config["n_layers"]):
            prefix = f"transformer.h.{i}"

            c_attn_weight = sd_hf[f"{prefix}.attn.c_attn.weight"]
            c_attn_bias = sd_hf[f"{prefix}.attn.c_attn.bias"]

            # attention proj
            sd[f"decoder_blocks.{i}.attn.to_qkv.weight"].copy_(c_attn_weight.t())
            sd[f"decoder_blocks.{i}.attn.to_qkv.bias"].copy_(c_attn_bias)
            
            # attention out
            sd[f"decoder_blocks.{i}.attn.to_out.weight"].copy_(
                sd_hf[f"{prefix}.attn.c_proj.weight"].t()
            )
            sd[f"decoder_blocks.{i}.attn.to_out.bias"].copy_(
                sd_hf[f"{prefix}.attn.c_proj.bias"]
            )
            
            # ffn
            sd[f"decoder_blocks.{i}.ffn.linear1.weight"].copy_(
                sd_hf[f"{prefix}.mlp.c_fc.weight"].t()
            )
            sd[f"decoder_blocks.{i}.ffn.linear1.bias"].copy_(
                sd_hf[f"{prefix}.mlp.c_fc.bias"]
            ) 

            sd[f"decoder_blocks.{i}.ffn.linear2.weight"].copy_(
                sd_hf[f"{prefix}.mlp.c_proj.weight"].t()
            )
            sd[f"decoder_blocks.{i}.ffn.linear2.bias"].copy_(
                sd_hf[f"{prefix}.mlp.c_proj.bias"]
            )

            # layernorm
            sd[f"decoder_blocks.{i}.ln1.weight"].copy_(
                sd_hf[f"{prefix}.ln_1.weight"]
            )
            sd[f"decoder_blocks.{i}.ln1.bias"].copy_(
                sd_hf[f"{prefix}.ln_1.bias"]
            )
            sd[f"decoder_blocks.{i}.ln2.weight"].copy_(
                sd_hf[f"{prefix}.ln_2.weight"]
            )
            sd[f"decoder_blocks.{i}.ln2.bias"].copy_(
                sd_hf[f"{prefix}.ln_2.bias"]
            )
        sd["layer_norm.weight"].copy_(sd_hf["transformer.ln_f.weight"])
        sd["layer_norm.bias"].copy_(sd_hf["transformer.ln_f.bias"])
        
        return model


    def clear_kv_cache(self):
        for decoder_block in self.decoder_blocks:
            decoder_block.attn.clear_kv_cache()

    def get_kv_cache_seqlen(self):
        return self.decoder_blocks[0].attn.get_kv_cache_seqlen()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0., std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0., std=0.02)
        

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
    
    def forward(self,
                x: Tensor,
                use_kv_cache: bool=True):
        causal_attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1)
        x = x + self.attn(self.ln1(x), causal_attn_mask, use_kv_cache)
        x = x + self.ffn(self.ln2(x))
        return x

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
    
model = GPT.from_pretrained("gpt2")