"""Self-checks for the Mixtral MoE layer: dense dispatch == masked (sparse) dispatch, values and gradients."""
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.models.mixtral.model import Mixtral, MOE

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()

    # 1. MoE layer alone, float64
    moe = MOE(32, 64, n_experts=4, n_experts_per_tok=2).to(device).double()
    x = torch.randn(3, 17, 32, device=device, dtype=torch.float64)
    outs, grads = {}, {}
    for dense in (False, True):
        moe.dense = dense
        moe.zero_grad()
        xg = x.clone().requires_grad_(True)
        out = moe(xg)
        out.square().sum().backward()
        outs[dense], grads[dense] = out, [xg.grad] + [p.grad.clone() for p in moe.parameters()]
    print(f"MoE layer | dense vs masked rel err {rel(outs[False], outs[True]):.1e} "
          f"| grad rel err {max(rel(a, b) for a, b in zip(grads[False], grads[True])):.1e}")

    # 2. full model, and causality
    torch.manual_seed(0)
    model = Mixtral(n_layers=2, n_heads=4, embed_dim=64, n_experts=4, n_experts_per_tok=2, vocab_size=65,
                    block_size=128, n_kv_heads=2, ffn_hidden_dim=128).to(device).eval()
    model.init_weights()
    idx = torch.randint(0, 65, (2, 100), device=device)
    with torch.no_grad():
        logits = {}
        for dense in (False, True):
            for blk in model.decoder_blocks:
                blk.moe.dense = dense
            logits[dense] = model(idx)[0]
        idx2 = idx.clone()
        idx2[:, 50:] = torch.randint(0, 65, idx2[:, 50:].shape, device=device)
        leak = (model(idx)[0][:, :50] - model(idx2)[0][:, :50]).abs().max().item()
    print(f"model | dense vs masked rel err {rel(logits[False], logits[True]):.1e} | causal leak {leak:.1e}")

    # 3. training-shape timing
    if device == "cuda":
        model = Mixtral(n_layers=6, n_heads=6, embed_dim=384, n_experts=4, n_experts_per_tok=2, vocab_size=65,
                        block_size=256, n_kv_heads=2, ffn_hidden_dim=512).to(device)
        xb = torch.randint(0, 65, (16, 256), device=device)
        for dense in (False, True):
            for blk in model.decoder_blocks:
                blk.moe.dense = dense
            for i in range(3):
                if i == 1:
                    torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xb)[0]
                out.float().square().mean().backward()
            torch.cuda.synchronize()
            print(f"{'dense ' if dense else 'masked'} dispatch, batch 16 x 256, 6 layers | {(time.time() - t) / 2 * 1000:.0f} ms/iter")
