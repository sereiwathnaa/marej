from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")

class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, (self.weight.shape[0], ), self.weight, self.bias, 1e-5)

class ScratchLayerNorm(nn.Module):
    def __init__(self, dim, bias: bool = True, eps: float=1e-5):
        super().__init__()

        if type(dim) is int:
            dim = (dim, )
        
        self.dim = tuple(dim)
        self.axis = tuple([-i for i in range(1, 1+len(dim))])
        self.eps = 1e-5

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else 0

    def forward(self, x):
        mean = x.mean(axis=self.axis, keepdims=True)
        var = x.var(axis=self.axis, keepdims=True)
        x_normalized = (x - mean) / (var + self.eps) ** 0.5
        x_normalized = x_normalized * self.weight + self.bias

        return x_normalized

class ScratchRMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, tuple[int]],
                 dim: Union[int, tuple[int]]=None,
                 eps: float=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = tuple(normalized_shape)

        self.dim = tuple([-i for i in range(1 , 1+len(normalized_shape))])
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        rms = (x ** 2).mean(dim=self.dim, keepdims=True)
        x_normalized = x / (rms + self.eps) ** 0.5
        x_normalized = x_normalized * self.weight
        return x_normalized