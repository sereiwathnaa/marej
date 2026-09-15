"""Scaled dot-product attention core (used by MultiheadAttention and GroupedQueryRotaryAttention)."""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange


class DotProductAttention(nn.Module):
    """Scaled dot-product attention over (batch, heads, seqlen, dim_head) tensors.

    attn_mask: (query seqlen, key seqlen); nonzero/True = NOT allowed to attend.
    Returns (batch, seqlen, heads * dim_head).
    """

    def __init__(self, use_flash: bool=True, dropout_p: float=0.):
        super().__init__()
        self.dropout = dropout_p
        self.use_flash = use_flash

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor=None):
        dropout_p = self.dropout if self.training else 0.
        if self.use_flash:
            if attn_mask is not None:
                out = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask.bool().logical_not(), dropout_p=dropout_p)
            else:
                out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout_p, is_causal=False)
        else:
            logits = torch.einsum("b h i d, b h j d -> b h i j", query, key) * query.shape[-1] ** -0.5
            if attn_mask is not None:
                logits = logits.masked_fill(attn_mask.bool(), value=-1e9)
            attn = F.softmax(logits, dim=-1)
            attn = F.dropout(attn, self.dropout, training=self.training)
            out = torch.einsum("b h i j, b h j d -> b h i d", attn, value)

        return rearrange(out, "b h l d -> b l (h d)")


if __name__ == "__main__":
    # Self-checks: flash vs manual softmax path, a float64 reference, and mask semantics.
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, h, l, d = 2, 4, 37, 16
    q, k, v = (torch.randn(b, h, l, d, device=device) for _ in range(3))
    causal_mask = torch.ones((l, l), device=device).triu(1)
    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()

    def reference(q, k, v, attn_mask):
        """Explicit float64 attention with the same mask convention (nonzero = not allowed)."""
        q, k, v = (t.double() for t in (q, k, v))
        logits = q @ k.transpose(-1, -2) * d ** -0.5
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask.bool(), float("-inf"))
        return rearrange(logits.softmax(-1) @ v, "b h l d -> b l (h d)")

    flash, manual = DotProductAttention(use_flash=True).eval(), DotProductAttention(use_flash=False).eval()
    for name, mask in (("no mask", None), ("causal mask", causal_mask)):
        ref = reference(q, k, v, mask)
        print(f"{name:12s} | flash vs reference rel err {rel(ref, flash(q, k, v, mask).double()):.2e} "
              f"| manual vs reference rel err {rel(ref, manual(q, k, v, mask).double()):.2e}")

    # masked keys must have no influence: perturb keys/values at positions >= t, outputs before t must not move
    t = l // 2
    k2, v2 = k.clone(), v.clone()
    k2[:, :, t:], v2[:, :, t:] = torch.randn_like(k2[:, :, t:]), torch.randn_like(v2[:, :, t:])
    for name, attn in (("flash", flash), ("manual", manual)):
        leak = (attn(q, k, v, causal_mask)[:, :t] - attn(q, k2, v2, causal_mask)[:, :t]).abs().max().item()
        print(f"{name:6s} | causal leak {leak:.1e}")
