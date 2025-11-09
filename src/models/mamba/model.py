#%%
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union
from dataclasses import dataclass

# %%
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int=16
    expand: int=2
    dt_rank: Union[int, str] = "auto"
    d_conv: int=4
    pad_vocab_size_multiple: int=8
    conv_bias: bool=True
    bias: bool=True

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple
#%%

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d_weight = nn.Parameter(torch.randn(args.d_inner, args.d_conv))
        self.conv1d_bias = nn.Parameter(torch.randn(args.d_inner))

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state *2, bias=False)

        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), "d_state -> d_inner d_state", d_inner=args.d_inner)
        self.A_log = nn.Paramter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        x, res = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv1d()
        pass
    
    def conv1d(self, x):
        b, l, d_in = x.shape
        padding = torch.zeros((b, self.args.d_conv - 1, d_in))
        x_padded = torch.cat([padding, x], dim=1)
        x_conv = torch.zeros_like(x) + self.conv1d_bias
        for i in range(self.args.d_conv):
            x_scaled = x_padded * self.conv1d_weight[:, i]
            x_scaled = x_scaled[:, i:l + i]
            x_conv = x_conv + x_scaled
        return x_conv