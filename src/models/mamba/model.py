from __future__ import annotations
import os
import sys
import math
import json
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, einsum

# project root, so `from src...` works whether run as a script, notebook, or module
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.layers.normalization import RMSNorm

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    # 'parallel': log2(L)-step Hillis-Steele scan with gradient checkpointing (fast, low memory; use for training)
    # 'sequential': token-by-token reference loop
    scan: str = 'parallel'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = self.embedding

    def forward(self, input_ids, use_kv_cache: bool = False):

        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = einsum(x, self.lm_head.weight, 'b l d, v d -> b l v')

        return logits

    def init_weights(self, dt_min: float=1e-3, dt_max: float=1e-1):
        """From-scratch init (not needed when loading pretrained weights).

        Embedding / linear weights ~ N(0, 0.02) as in GPT-2 (the embedding is tied to the LM head, so the
        default N(0, 1) init gives logits of order sqrt(d_model)); out_proj scaled by 1/sqrt(2 n_layer) for
        the residual stream; dt_proj initialised as in the Mamba reference so softplus(dt_proj bias) is
        log-uniform in [dt_min, dt_max].
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)
        for layer in self.layers:
            mixer = layer.mixer
            nn.init.normal_(mixer.out_proj.weight, std=0.02 / math.sqrt(2 * self.args.n_layer))
            dt_init_std = self.args.dt_rank ** -0.5
            nn.init.uniform_(mixer.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(torch.rand(self.args.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
            dt = dt.clamp(min=1e-4)
            with torch.no_grad():
                mixer.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
    
class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        
    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)
    
        Returns:
            output: shape (b, l, d)

        """
        output = self.mixer(self.norm(x)) + x
        return output
            

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        self.conv1d = nn.Conv1d(
            args.d_inner, args.d_inner, args.d_conv, bias=args.conv_bias, groups=args.d_inner, padding=args.d_conv - 1
        )

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner) 
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        
    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)
    
        Returns:
            output: shape (b, l, d)
        
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]  # Causal convolution
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)
        
        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):

        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t ;  y_t = C_t . h_t + D u_t

        Shapes: u, delta (b, l, d_in); A (d_in, n); B, C (b, l, n); D (d_in). Returns (b, l, d_in).
        """
        delta = delta.unsqueeze(-1)
        
        # deltaA: (b, l, d_in, n)
        deltaA = torch.exp(delta * A)
        
        # deltaB_u: (b, l, d_in, n)
        deltaB_u = delta * B.unsqueeze(2) * u.unsqueeze(-1)

        if self.args.scan == 'parallel':
            # recompute the scan in backward instead of storing log2(L) full-size intermediates
            h = checkpoint(self.parallel_scan, deltaA, deltaB_u, use_reentrant=False)
        else:
            h = self.sequential_scan(deltaA, deltaB_u)

        y = einsum(h, C, 'b l d n, b l n -> b l d')
        y = y + u * D

        return y

    @staticmethod
    def sequential_scan(a, b):
        """Reference: h_t = a_t * h_{t-1} + b_t, one step at a time. a, b, out: (b, l, d_in, n)."""
        h = torch.zeros_like(b[:, 0])
        hs = []
        for i in range(a.shape[1]):
            h = a[:, i] * h + b[:, i]
            hs.append(h)
        return torch.stack(hs, dim=1)

    @staticmethod
    def parallel_scan(a, b, chunk: int=16):
        """Same recurrence with a chunked Hillis-Steele scan: log2(chunk) rounds of whole-tensor ops inside each
        chunk, then a cheap sequential carry across the L/chunk chunk boundaries.

        (a_t, b_t) represents the affine map h -> a_t h + b_t of one step; composing the maps of steps
        [t-k, t] with [t-2k, t-k) doubles the span each round, so after the rounds a_t is the cumulative
        product from the chunk start and b_t is the state assuming h = 0 at the chunk start.
        """
        bsz, L, d, n = a.shape
        pad = (-L) % chunk
        if pad:
            a = F.pad(a, (0, 0, 0, 0, 0, pad), value=1.)   # identity steps at the end
            b = F.pad(b, (0, 0, 0, 0, 0, pad), value=0.)
        a = a.reshape(bsz, -1, chunk, d, n)
        b = b.reshape(bsz, -1, chunk, d, n)

        # roll + where instead of pad/slice: their backward passes are rolls and masks, no zero-filled copies
        pos = torch.arange(chunk, device=a.device).view(1, 1, chunk, 1, 1)
        k = 1
        while k < chunk:
            valid = pos >= k                                                    # positions with a step k back
            a_prev = torch.where(valid, a.roll(k, dims=2), 1.)                  # identity map where none
            b_prev = torch.where(valid, b.roll(k, dims=2), 0.)
            b = b + a * b_prev
            a = a * a_prev
            k *= 2

        # carry the state across chunks: h_end(j) = b_end(j) + a_end(j) * h_end(j-1)
        a_end, b_end = a[:, :, -1].unbind(1), b[:, :, -1].unbind(1)           # slice once, not per chunk
        carry = torch.zeros_like(b_end[0])
        carries = []
        for j in range(len(a_end)):
            carries.append(carry)
            carry = b_end[j] + a_end[j] * carry
        carries = torch.stack(carries, dim=1).unsqueeze(2)                 # (bsz, n_chunks, 1, d, n)

        h = (b + a * carries).reshape(bsz, -1, d, n)
        return h[:, :L]


def load_pretrained_mamba(pretrained_model_name: str, device=None):
    """Load pretrained weights from HuggingFace into PyTorch Mamba model.
    
    Args:
        pretrained_model_name: One of
            * 'state-spaces/mamba-2.8b-slimpj'
            * 'state-spaces/mamba-2.8b'
            * 'state-spaces/mamba-1.4b'
            * 'state-spaces/mamba-790m'
            * 'state-spaces/mamba-370m'
            * 'state-spaces/mamba-130m'
        device: Device to load model on (default: None, stays on CPU)
                        
    Returns:
        model: Mamba model with pretrained weights loaded
    """
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    def load_config_hf(model_name):
        resolved_archive_file = cached_file(
            model_name, 
            CONFIG_NAME,
            _raise_exceptions_for_missing_entries=False
        )
        return json.load(open(resolved_archive_file))
    
    def load_state_dict_hf(model_name):
        resolved_archive_file = cached_file(
            model_name, 
            WEIGHTS_NAME,
            _raise_exceptions_for_missing_entries=False
        )
        return torch.load(
            resolved_archive_file, 
            weights_only=True, 
            map_location='cpu', 
            mmap=True
        )
    
    # Load config and create model
    config_data = load_config_hf(pretrained_model_name)
    args = ModelArgs(
        d_model=config_data['d_model'],
        n_layer=config_data['n_layer'],
        vocab_size=config_data['vocab_size']
    )
    model = Mamba(args)
    
    # Load HuggingFace state dict
    state_dict = load_state_dict_hf(pretrained_model_name)
    
    def load_tensor(name):
        """Pop tensor from state_dict and convert to float."""
        return state_dict.pop(name).float()
    
    # Get model's named parameters for direct assignment
    model_params = dict(model.named_parameters())
    
    # Load embedding weights (tied with lm_head)
    model_params['embedding.weight'].data.copy_(
        load_tensor('backbone.embedding.weight')
    )
    
    # Load final norm weights
    model_params['norm_f.weight'].data.copy_(
        load_tensor('backbone.norm_f.weight')
    )
    
    for layer_i in range(args.n_layer):
        prefix = f'backbone.layers.{layer_i}'
        model_prefix = f'layers.{layer_i}'
        
        in_proj_weight = load_tensor(f'{prefix}.mixer.in_proj.weight')
        model_params[f'{model_prefix}.mixer.in_proj.weight'].data.copy_(in_proj_weight)
        
        if f'{prefix}.mixer.in_proj.bias' in state_dict:
            in_proj_bias = load_tensor(f'{prefix}.mixer.in_proj.bias')
            model_params[f'{model_prefix}.mixer.in_proj.bias'].data.copy_(in_proj_bias)
        
        # Conv1d weights: HF and our depthwise nn.Conv1d both use (d_inner, 1, d_conv)
        model_params[f'{model_prefix}.mixer.conv1d.weight'].data.copy_(
            load_tensor(f'{prefix}.mixer.conv1d.weight')
        )
        model_params[f'{model_prefix}.mixer.conv1d.bias'].data.copy_(
            load_tensor(f'{prefix}.mixer.conv1d.bias')
        )
        
        # x_proj (projects to dt_rank + 2*d_state)
        x_proj_weight = load_tensor(f'{prefix}.mixer.x_proj.weight')
        model_params[f'{model_prefix}.mixer.x_proj.weight'].data.copy_(x_proj_weight)
        
        # dt_proj
        dt_proj_weight = load_tensor(f'{prefix}.mixer.dt_proj.weight')
        model_params[f'{model_prefix}.mixer.dt_proj.weight'].data.copy_(dt_proj_weight)
        model_params[f'{model_prefix}.mixer.dt_proj.bias'].data.copy_(
            load_tensor(f'{prefix}.mixer.dt_proj.bias')
        )
        
        # A_log and D (state space parameters)
        model_params[f'{model_prefix}.mixer.A_log'].data.copy_(
            load_tensor(f'{prefix}.mixer.A_log')
        )
        model_params[f'{model_prefix}.mixer.D'].data.copy_(
            load_tensor(f'{prefix}.mixer.D')
        )
        
        # out_proj
        out_proj_weight = load_tensor(f'{prefix}.mixer.out_proj.weight')
        model_params[f'{model_prefix}.mixer.out_proj.weight'].data.copy_(out_proj_weight)
        
        # Check if out_proj has bias
        if f'{prefix}.mixer.out_proj.bias' in state_dict:
            out_proj_bias = load_tensor(f'{prefix}.mixer.out_proj.bias')
            model_params[f'{model_prefix}.mixer.out_proj.bias'].data.copy_(out_proj_bias)
        
        # Layer norm
        model_params[f'{model_prefix}.norm.weight'].data.copy_(
            load_tensor(f'{prefix}.norm.weight')
        )
    
    # Check if all weights were loaded
    if state_dict:
        print(f"Warning: The following weights were not loaded: {list(state_dict.keys())}")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
    
    print(f"✅ Successfully loaded pretrained weights from {pretrained_model_name}")
    return model


# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    # Load a pretrained model
    model = load_pretrained_mamba('state-spaces/mamba-130m')
    
    # Test the model
    batch_size = 2
    seq_len = 10
    vocab_size = model.args.vocab_size
    
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model device: {next(model.parameters()).device}")

    def batch_generation_example():
        """Example of generating multiple tokens."""
        print("\n🔄 Batch generation example...")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
        model = load_pretrained_mamba('state-spaces/mamba-130m')
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Prepare input
        prompt = "The weather today is very"
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        
        print(f"📝 Prompt: {prompt}")
        
        # Generate multiple tokens
        model.eval()
        generated_tokens = []
        
        with torch.no_grad():
            for i in range(20):  # Generate 20 tokens
                logits = model(input_ids)
                next_token_logits = logits[:, -1, :]
                
                # Simple greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens)
        print(f"💬 Generated: {generated_text}")
        print(f"📄 Full text: {prompt}{generated_text}")
    batch_generation_example()
