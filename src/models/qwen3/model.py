import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple
from einops import repeat
from src.layers.attention import GroupedQueryRotaryAttention
from src.layers.normalization import RMSNorm
import math

USE_BASE_MODEL = False
USE_REASONINING_MODEL = True
USE_INSTRUCT_MODEL = False


