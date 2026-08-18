import os
import sys
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# project root, so `from src...` works whether run as a script, notebook, or module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.layers.attention import GroupedQueryRotaryAttention
from src.layers.normalization import RMSNorm

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



class DecoderBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 ffn_hidden_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 block_size: int,
                 rotary_base: int,
                 norm_eps: float,
                 use_flash: bool,
                 ):
        super().__init__()

        self.norm1 = RMSNorm(embed_dim, eps=norm_eps)
        self.attn = GroupedQueryRotaryAttention(
            embed_dim, n_heads, n_kv_heads,
            dropout_p=0., apply_rotary_embedding=True,
            rotary_base=rotary_base, max_seqlen=block_size,
            bias=False, use_flash=use_flash,
            batch_first=True
        )
        self.norm2 = RMSNorm(embed_dim, eps=norm_eps)
        self.ffn = FeedForwardBlock(embed_dim, ffn_hidden_dim)

    def forward(self,
                x: Tensor,
                use_kv_cache: bool,
                rotation_matr: Tuple[Tensor, Tensor]=None):
        x = x + self.self_attn(self.norm1(x), use_kv_cache, rotation_matr)
        x = x + self.ffn(self.norm2(x))
        return x

    def self_attn(self,
                  x: Tensor,
                  use_kv_cache: bool,
                  rotation_matr: Tuple[Tensor, Tensor]=None):
        causal_attn_mask = torch.ones((x.shape[1], x.shape[1]), device=x.device).triu(1)
        out = self.attn(x, causal_attn_mask, use_kv_cache, rotation_matr)
        return out


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
                 norm_eps: float=1e-5,
                 use_flash: bool=True):
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
            [DecoderBlock(embed_dim, ffn_hidden_dim, n_heads, n_kv_heads, block_size, rotary_base, norm_eps, use_flash) for _ in range(n_layers)]
        )
        
        self.rms_norm = RMSNorm(embed_dim, eps=norm_eps)
        # No tying weight like palm, gpt, llama
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        self.rotation_matr = self.decoder_blocks[0].attn.compute_rotation_matrix()

    def forward(self,
                indices: Tensor,
                use_kv_cache: bool=False):
        if self.rotation_matr[0].device != indices.device:
            self.rotation_matr = tuple(t.to(indices.device) for t in self.rotation_matr)
        x = self.word_embeddings(indices)
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, use_kv_cache, self.rotation_matr)

        x = self.rms_norm(x)
        logits = self.output_projection(x)

        return logits

    @staticmethod
    def from_pretrained(model_name: str,
                        model_dir: str=None):
        from transformers import AutoModelForCausalLM
        model_hf = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir)
        config_hf = model_hf.config

        config = {
            "n_layers": config_hf.num_hidden_layers,
            "n_heads": config_hf.num_attention_heads,
            "embed_dim": config_hf.hidden_size,
            "vocab_size": config_hf.vocab_size,
            "block_size": config_hf.max_position_embeddings,
            "n_kv_heads": config_hf.num_key_value_heads,
            "ffn_hidden_dim": config_hf.intermediate_size,
            "rotary_base": getattr(config_hf, "rope_theta", 10000),
            "norm_eps": config_hf.rms_norm_eps,
            "use_flash": True,
        }

        model = Llama(**config)

        sd_hf = model_hf.state_dict()
        sd = model.state_dict()

        sd["word_embeddings.weight"].copy_(sd_hf["model.embed_tokens.weight"])
        sd["output_projection.weight"].copy_(sd_hf["lm_head.weight"])
        sd["rms_norm.weight"].copy_(sd_hf["model.norm.weight"])

        for i in range(config["n_layers"]):
            prefix = f"model.layers.{i}"
            block_prefix = f"decoder_blocks.{i}"

            sd[f"{block_prefix}.norm1.weight"].copy_(
                sd_hf[f"{prefix}.input_layernorm.weight"]
            )
            sd[f"{block_prefix}.norm2.weight"].copy_(
                sd_hf[f"{prefix}.post_attention_layernorm.weight"]
            )

            sd[f"{block_prefix}.attn.to_q.weight"].copy_(
                sd_hf[f"{prefix}.self_attn.q_proj.weight"]
            )

            k_weight = sd_hf[f"{prefix}.self_attn.k_proj.weight"]
            v_weight = sd_hf[f"{prefix}.self_attn.v_proj.weight"]
            sd[f"{block_prefix}.attn.to_kv.weight"].copy_(torch.cat([k_weight, v_weight], dim=0))

            sd[f"{block_prefix}.attn.to_out.weight"].copy_(
                sd_hf[f"{prefix}.self_attn.o_proj.weight"]
            )

            sd[f"{block_prefix}.ffn.linear1.weight"].copy_(
                sd_hf[f"{prefix}.mlp.gate_proj.weight"]
            )
            sd[f"{block_prefix}.ffn.linear2.weight"].copy_(
                sd_hf[f"{prefix}.mlp.down_proj.weight"]
            )
            sd[f"{block_prefix}.ffn.linear3.weight"].copy_(
                sd_hf[f"{prefix}.mlp.up_proj.weight"]
            )

        return model

