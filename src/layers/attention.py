"""Backward-compatible re-exports. Each attention type now lives in its own module:

    dotproductattention.py   DotProductAttention
    multiheadattention.py    MultiheadAttention
    groupedqueryattention.py GroupedQueryRotaryAttention
    sparseattention.py       FixedSparseAttention, StridedSparseAttention
    deltaattention.py        KimiDeltaAttention, GatedDeltaRule

Run any of them as a module (e.g. `python -m src.layers.sparseattention`) for its self-checks.
"""

from .dotproductattention import DotProductAttention
from .multiheadattention import MultiheadAttention
from .groupedqueryattention import GroupedQueryRotaryAttention
from .sparseattention import FixedSparseAttention, StridedSparseAttention
from .deltaattention import KimiDeltaAttention, GatedDeltaRule

__all__ = ["DotProductAttention", "MultiheadAttention", "GroupedQueryRotaryAttention",
           "FixedSparseAttention", "StridedSparseAttention", "KimiDeltaAttention", "GatedDeltaRule"]
