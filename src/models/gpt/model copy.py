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

        # self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        # self.position_embeddings = nn.Embedding(block_size, embed_dim)

        # self.decoder_blocks = nn.ModuleList(
        #     [DecoderBlock(embed_dim, n_heads, self.dim_head, dropout_p) for _ in range(n_layers)]
        # )
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, embed_dim),
            wpe = nn.Embedding(block_size, embed_dim),
            drop = nn.Dropout(dropout_p),
            h = nn.ModuleList([DecoderBlock(embed_dim, n_heads, self.dim_head, dropout_p=dropout_p) for _ in range(n_layers)]),
            ln_f = LayerNorm(embed_dim),
        ))

        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight


    def forward(self,
                indices: Tensor,
                use_kv_cache: bool=True,
                targets: Tensor=None,
                ):
        offset = self.get_kv_cache_seqlen() if use_kv_cache else 0
        position_indices = torch.arange(indices.shape[1], device=indices.device) + offset
        x = self.transformer.wte(indices) + self.transformer.wpe(position_indices).unsqueeze(0)
        x = self.dropout(x)

        for decoder_block in self.transformer.h:
            x = decoder_block(x, use_kv_cache)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
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
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


    def clear_kv_cache(self):
        for decoder_block in self.transformer.h:
            decoder_block.attn.clear_kv_cache()

    def get_kv_cache_seqlen(self):
        return self.transformer.h[0].attn.get_kv_cache_seqlen()

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

        self.ln_1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, n_heads, dim_head, dropout_p, use_flash=False)
        self.ln_2 = LayerNorm(embed_dim)
        self.mlp = FeedForwardBlock(embed_dim, embed_dim * 4)
    
    def forward(self,
                x: Tensor,
                use_kv_cache: bool=True):
        causal_attn_mask = torch.triu(torch.ones(x.shape[1], x.shape[1], device=x.device), diagonal=1)
        x = x + self.attn(self.ln_1(x), causal_attn_mask, use_kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super().__init__()

        self.c_fc = nn.Linear(input_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x
    
model = GPT.from_pretrained("gpt2")