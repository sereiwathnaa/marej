"""Self-checks for the Mamba block: parallel scan == sequential reference (values and gradients)."""
import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.models.mamba.model import Mamba, MambaBlock, ModelArgs

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()

    # 1. scan primitive, including a non-power-of-two length and strong decays
    for l in (37, 128):
        a = torch.rand(2, l, 24, 8, device=device, dtype=torch.float64) ** 4
        b = torch.randn(2, l, 24, 8, device=device, dtype=torch.float64)
        print(f"scan L={l} | parallel vs sequential rel err {rel(MambaBlock.sequential_scan(a, b), MambaBlock.parallel_scan(a, b)):.1e}")

    # 2. full model: outputs and gradients, parallel (checkpointed) vs sequential
    x = torch.randint(0, 65, (2, 100), device=device)
    outs, grads = {}, {}
    for scan in ("sequential", "parallel"):
        torch.manual_seed(0)
        model = Mamba(ModelArgs(d_model=64, n_layer=2, vocab_size=65, scan=scan)).to(device)
        logits = model(x)
        logits.square().mean().backward()
        outs[scan] = logits
        grads[scan] = [p.grad for p in model.parameters()]
    print(f"model | logits rel err {rel(outs['sequential'], outs['parallel']):.1e} "
          f"| grad rel err {max(rel(g1, g2) for g1, g2 in zip(grads['sequential'], grads['parallel'])):.1e}")

    # 3. causality: changing tokens at positions >= t must not change logits before t
    model.eval()
    with torch.no_grad():
        x2 = x.clone()
        x2[:, 50:] = torch.randint(0, 65, x2[:, 50:].shape, device=device)
        print(f"causal leak | {(model(x)[:, :50] - model(x2)[:, :50]).abs().max():.1e}")

    # 4. training-shape timing
    if device == "cuda":
        for scan in ("sequential", "parallel"):
            model = Mamba(ModelArgs(d_model=384, n_layer=6, vocab_size=65, scan=scan)).to(device)
            xb = torch.randint(0, 65, (32, 128), device=device)
            torch.cuda.reset_peak_memory_stats()
            for i in range(3):
                if i == 1:
                    torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                logits.float().square().mean().backward()
            torch.cuda.synchronize()
            print(f"{scan:10s} scan, batch 32 x 128, 6 layers | {(time.time() - t) / 2 * 1000:.0f} ms/iter "
                  f"| peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
