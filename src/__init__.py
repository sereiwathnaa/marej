import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  

from . import nlp
from . import models

from .layers import (
    MultiheadAttention,
    LayerNorm
)