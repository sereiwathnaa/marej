import os
import sys
import inspect
from dataclasses import dataclass

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# project root, so `from src...` works whether run as a script, notebook, or module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.layers.normalization import LayerNorm
from src.layers.attention import MultiheadAttention

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    embed_dim: int = 768
    dropout_p: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 vocab_size: int,
                 block_size: int,
                 dropout_p: float,
                 bias: bool=True):
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

        self.layer_norm = LayerNorm(embed_dim, bias=bias)
        # weight tying: the LM head reuses the token embedding matrix
        self.output_projection = self.word_embeddings.weight

        self.apply(self._init_weights)

    def forward(self,
                indices: Tensor,
                use_kv_cache: bool=False,
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
    
    @torch.no_grad()
    def generate_sample(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self.forward(idx_cond, targets=None, use_kv_cache=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @staticmethod
    def from_pretrained(model_name: str, dropout_p: float=None):
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
            'dropout_p': model_hf.transformer.drop.p if dropout_p is None else dropout_p,
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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.word_embeddings.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = self.n_layers, self.n_heads, self.embed_dim//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    
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
                use_kv_cache: bool=False):
        causal_attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1)
        x = x + self.dropout(self.attn(self.ln1(x), causal_attn_mask, use_kv_cache))
        x = x + self.dropout(self.ffn(self.ln2(x)))
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
