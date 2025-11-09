
#%%
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
import math
from dataclasses import dataclass
from typing import Union
import sys
sys.path.append("/home/nyxx/my_project/marejv2/")
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

        # Tie output projection to embedding weights (Weight Tying)
        self.lm_head = self.embedding

    def forward(self, input_ids, use_kv_cache: bool = False):
        """
        Args:
            input_ids (long tensor): shape (b, l)
            use_kv_cache (bool): dummy parameter to maintain compatibility with generation scripts.
    
        Returns:
            logits: shape (b, l, vocab_size)

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = einsum(x, self.lm_head.weight, 'b l d, v d -> b l v')

        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from .loadpretrained import load_pretrained_mamba
        return load_pretrained_mamba(pretrained_model_name)
        

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
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

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

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
        """Runs the SSM. See Algorithm 2 in Section 3.2 in the Mamba paper [1]

        Args:
            x: shape (b, l, d_in)
    
        Returns:
            output: shape (b, l, d_in)

        """
        (d_in, n) = self.A_log.shape
        
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        delta = delta.unsqueeze(-1)
        
        # deltaA: (b, l, d_in, n)
        deltaA = torch.exp(delta * A)
        
        # deltaB_u: (b, l, d_in, n)
        deltaB_u = delta * B.unsqueeze(2) * u.unsqueeze(-1)
        
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.sum(x * C[:, i:i+1, :], dim=-1)
            ys.append(y)
        
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D

        return y
# %%
args = ModelArgs(
    d_model=128,      # embedding dimension
    n_layer=4,        # number of layers
    vocab_size=1000,  # vocabulary size
    d_state=16,       # state dimension for SSM
    expand=2,         # expansion factor
    dt_rank="auto",   # automatically set dt_rank
    d_conv=4,         # convolution kernel size
    pad_vocab_size_multiple=8,
    conv_bias=True,
    bias=True
)

# Instantiate the Mamba model
model = Mamba(args)

# Create dummy input: batch_size=2, seq_len=10, with random token indices
batch_size = 2
seq_len = 10
dummy_indices = torch.randint(0, args.vocab_size, (batch_size, seq_len))

# Forward pass
with torch.no_grad():
    logits = model.forward(dummy_indices)

print(f"Input shape: {dummy_indices.shape}")
print(f"Output logits shape: {logits.shape}") 
# %%
