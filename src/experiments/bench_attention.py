"""Train a tiny GPT on shakespeare_char with every attention variant swapped in, and sanity-check each."""
import os, sys, time, pickle, json
import numpy as np
import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)
from src.models.gpt.model import GPT
from src.layers.attention import (MultiheadAttention, GroupedQueryRotaryAttention,
                                  FixedSparseAttention, StridedSparseAttention)
from src.layers.deltaattention import KimiDeltaAttention
from src.nlp.generation import batch_generation

device = "cuda"
# TF32 off so the fp32 consistency checks are exact; training uses bf16 autocast anyway
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---- config (baby model so every variant trains in ~1 min) ----
N_LAYERS, N_HEADS, EMBED, BLOCK = 4, 4, 128, 128
DIM_HEAD = EMBED // N_HEADS
BATCH, ITERS, LR = 64, 500, 1e-3
EVAL_EVERY, EVAL_ITERS = 100, 40
SPARSE_BLOCK = 16

data_dir = os.path.join(ROOT, "data", "shakespeare_char")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
meta = pickle.load(open(os.path.join(data_dir, "meta.pkl"), "rb"))
VOCAB, itos, stoi = meta["vocab_size"], meta["itos"], meta["stoi"]
decode = lambda ids: "".join(itos[i] for i in ids)


def get_batch(split, gen):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK, (BATCH,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


class SparseAdapter(nn.Module):
    """Fixed/StridedSparseAttention only take x; adapt to the (x, attn_mask, use_kv_cache) interface."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.kv_cache = None

    def forward(self, x, attn_mask=None, use_kv_cache=False):
        if use_kv_cache:
            raise NotImplementedError("sparse attention has no KV cache")
        return self.inner(x)

    def get_kv_cache_seqlen(self): return 0
    def clear_kv_cache(self): pass


VARIANTS = {
    "MultiheadAttention (flash)":            lambda: MultiheadAttention(EMBED, N_HEADS, DIM_HEAD, 0., use_flash=True),
    "MultiheadAttention (manual softmax)":   lambda: MultiheadAttention(EMBED, N_HEADS, DIM_HEAD, 0., use_flash=False),
    "GQRA no-rotary (== MHA)":               lambda: GroupedQueryRotaryAttention(EMBED, N_HEADS, N_HEADS, 0., apply_rotary_embedding=False, max_seqlen=BLOCK),
    "GQRA rotary":                           lambda: GroupedQueryRotaryAttention(EMBED, N_HEADS, N_HEADS, 0., apply_rotary_embedding=True, max_seqlen=BLOCK),
    "GQRA rotary, 2 kv heads, qk_norm":      lambda: GroupedQueryRotaryAttention(EMBED, N_HEADS, 2, 0., apply_rotary_embedding=True, max_seqlen=BLOCK, qk_norm=True),
    "GQRA rotary, manual softmax":           lambda: GroupedQueryRotaryAttention(EMBED, N_HEADS, N_HEADS, 0., apply_rotary_embedding=True, max_seqlen=BLOCK, use_flash=False),
    f"FixedSparseAttention (block {SPARSE_BLOCK})":   lambda: SparseAdapter(FixedSparseAttention(EMBED, N_HEADS, DIM_HEAD, block_size=SPARSE_BLOCK)),
    f"StridedSparseAttention (block {SPARSE_BLOCK})": lambda: SparseAdapter(StridedSparseAttention(EMBED, N_HEADS, DIM_HEAD, block_size=SPARSE_BLOCK)),
    "KimiDeltaAttention (chunk 32)":         lambda: KimiDeltaAttention(EMBED, N_HEADS, chunk_size=32),
}


def build_model(make_attn):
    torch.manual_seed(0)
    model = GPT(n_layers=N_LAYERS, n_heads=N_HEADS, embed_dim=EMBED, vocab_size=VOCAB, block_size=BLOCK, dropout_p=0.)
    for blk in model.decoder_blocks:
        blk.attn = make_attn()
        blk.attn.apply(model._init_weights)
    return model.to(device)


@torch.no_grad()
def estimate_loss(model, split):
    model.eval()
    gen = torch.Generator().manual_seed(123)
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(split, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def causality_leak(model):
    """Max change in logits at positions < t when tokens at positions >= t are replaced."""
    model.eval()
    gen = torch.Generator().manual_seed(7)
    x, _ = get_batch("val", gen)
    t = BLOCK // 2 + 3
    x2 = x.clone()
    x2[:, t:] = torch.randint(0, VOCAB, x2[:, t:].shape, generator=gen).to(device)
    return (model(x)[:, :t] - model(x2)[:, :t]).abs().max().item()


@torch.no_grad()
def kv_cache_consistency(model):
    """Max |logits| difference between one full forward and prefill + token-by-token cached decoding."""
    model.eval()
    gen = torch.Generator().manual_seed(11)
    x, _ = get_batch("val", gen)
    x = x[:4, :96]
    full = model(x)
    model.clear_kv_cache()
    parts = [model(x[:, :60], use_kv_cache=True)]
    for t in range(60, x.shape[1]):
        parts.append(model(x[:, t:t + 1], use_kv_cache=True))
    model.clear_kv_cache()
    return (full - torch.cat(parts, dim=1)).abs().max().item()


@torch.no_grad()
def sample(model, use_kv_cache):
    model.eval()
    prompt = torch.tensor([[stoi[c] for c in "ROMEO:\n"]], device=device)
    model.clear_kv_cache()
    torch.manual_seed(0)
    if use_kv_cache:
        toks = [t[0] for t in batch_generation(model, prompt, 120, top_k=20, temperature=0.8, use_kv_cache=True)]
    else:
        # sparse attention needs seqlen % SPARSE_BLOCK == 0: right-pad (causal, so padding cannot leak backwards)
        idx = prompt
        for _ in range(120):
            pad = (-idx.shape[1]) % SPARSE_BLOCK
            logits = model(torch.nn.functional.pad(idx, (0, pad)))[:, idx.shape[1] - 1] / 0.8
            v, _ = torch.topk(logits, 20)
            logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(torch.softmax(logits, -1), 1)], dim=1)
        toks = idx[0, prompt.shape[1]:].tolist()
    model.clear_kv_cache()
    return decode(toks)


# optional CLI filter: python bench_attention.py Kimi rotary   -> only variants whose name contains one of these
if len(sys.argv) > 1:
    VARIANTS = {n: f for n, f in VARIANTS.items() if any(a.lower() in n.lower() for a in sys.argv[1:])}

results = {}
for name, make_attn in VARIANTS.items():
    print(f"\n=== {name} ===", flush=True)
    model = build_model(make_attn)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    model.train()
    torch.cuda.synchronize(); t0 = time.time()
    history = []
    for it in range(1, ITERS + 1):
        x, y = get_batch("train", gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if it % EVAL_EVERY == 0:
            torch.cuda.synchronize()
            history.append((it, loss.item(), estimate_loss(model, "val")))
            print(f"  iter {it:4d} | train {loss.item():.3f} | val {history[-1][2]:.3f} | {(time.time() - t0) / it * 1000:.1f} ms/iter", flush=True)
    torch.cuda.synchronize()
    ms_per_iter = (time.time() - t0) / ITERS * 1000

    res = {"params": n_params, "ms_per_iter": ms_per_iter,
           "train_loss": estimate_loss(model, "train"), "val_loss": estimate_loss(model, "val"),
           "causal_leak": causality_leak(model), "history": history}
    supports_cache = not isinstance(model.decoder_blocks[0].attn, SparseAdapter)
    res["kv_cache_err"] = kv_cache_consistency(model) if supports_cache else None
    res["sample"] = sample(model, use_kv_cache=supports_cache)
    print(f"  final: train {res['train_loss']:.3f} | val {res['val_loss']:.3f} | causal leak {res['causal_leak']:.1e}"
          f" | kv-cache err {res['kv_cache_err']} | {ms_per_iter:.1f} ms/iter | params {n_params:,}")
    print("  sample: " + repr(res["sample"][:100]))
    results[name] = res
    del model, opt; torch.cuda.empty_cache()

json.dump(results, open(os.path.join(os.path.dirname(__file__), "bench_results.json"), "w"), indent=1)
print("\n" + "=" * 100)
print(f"{'variant':40s} {'val loss':>9s} {'train':>7s} {'leak':>8s} {'kv err':>8s} {'ms/iter':>8s}")
for name, r in results.items():
    kv = "n/a" if r["kv_cache_err"] is None else f"{r['kv_cache_err']:.1e}"
    print(f"{name:40s} {r['val_loss']:9.3f} {r['train_loss']:7.3f} {r['causal_leak']:8.1e} {kv:>8s} {r['ms_per_iter']:8.1f}")
