"""Triton kernels for the intra-chunk part of the gated delta rule with per-channel decay (KDA).

The chunkwise algorithm in `deltaattention.py` needs, for every chunk of length C, the lower-triangular
"decayed scores"

    S[i, j] = sum_d x[i, d] * y[j, d] * exp(gc[i, d] - gc[j, d])      for j <= i  (or j < i if strict)

where gc is the within-chunk cumulative log-decay (non-increasing along the sequence, so the ratio is <= 1
whenever j <= i). Materialising exp(gc_i - gc_j) costs O(C * C * d) memory per chunk and dominated the
PyTorch implementation. These kernels compute S and its gradients without that tensor:

* The chunk is tiled into BC x BC sub-blocks. For a sub-block pair (I, J) with J < I, pick an anchor row m
  with j <= m <= i for all i in I, j in J. Then exp(gc_i - gc_j) = exp(gc_i - gc_m) * exp(gc_m - gc_j) and
  both factors are <= 1, so the block is a plain dot product of decay-weighted rows (no overflow).
* Diagonal sub-blocks (I == J) are handled row by row with an explicit j <= i mask.

Usage: `intra_chunk_scores(x, y, gc, strict)` with x, y, gc of shape (N, C, K) in float32 on CUDA; returns
S of shape (N, C, C) and is differentiable w.r.t. x, y and gc. `intra_chunk_scores_torch` is the reference.
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def intra_chunk_scores_torch(x: Tensor, y: Tensor, gc: Tensor, strict: bool) -> Tensor:
    """Reference implementation (materialises the (N, C, C, K) pairwise decay tensor)."""
    C = x.shape[1]
    mask = torch.ones((C, C), dtype=torch.bool, device=x.device).tril(-1 if strict else 0)
    pair_decay = (gc.unsqueeze(-2) - gc.unsqueeze(-3)).masked_fill(~mask[:, :, None], float("-inf")).exp()
    return torch.einsum("n i d, n i j d, n j d -> n i j", x, pair_decay, y)


if HAS_TRITON:

    @triton.jit
    def _scores_fwd_kernel(x_ptr, y_ptr, g_ptr, s_ptr,
                           K: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
                           STRICT: tl.constexpr):
        # One program per (chunk, row sub-block I): writes S[I, :] for columns j <= i.
        pid_c = tl.program_id(0)
        pid_i = tl.program_id(1)
        base = pid_c * C * K
        s_base = pid_c * C * C
        o_k = tl.arange(0, BK)
        m_k = o_k < K
        o_r = tl.arange(0, BC)
        rows_I = pid_i * BC + o_r

        x_I = tl.load(x_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        g_I = tl.load(g_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        # anchor = first row of block I (all j in blocks J < I are <= it, all i in I are >= it)
        g_n = tl.load(g_ptr + base + (pid_i * BC) * K + o_k, mask=m_k, other=0.)
        xd = x_I * tl.exp(g_I - g_n[None, :])                                            # (BC, BK), factor <= 1

        for J in range(0, pid_i):
            rows_J = J * BC + o_r
            y_J = tl.load(y_ptr + base + rows_J[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
            g_J = tl.load(g_ptr + base + rows_J[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
            yd = y_J * tl.exp(g_n[None, :] - g_J)                                        # (BC, BK), factor <= 1
            s = tl.dot(xd, tl.trans(yd), allow_tf32=False)                               # (BC, BC)
            tl.store(s_ptr + s_base + rows_I[:, None] * C + rows_J[None, :], s)

        # diagonal sub-block, row by row with the causal mask
        y_I = tl.load(y_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        for r in range(0, BC):
            row = pid_i * BC + r
            x_r = tl.load(x_ptr + base + row * K + o_k, mask=m_k, other=0.)
            g_r = tl.load(g_ptr + base + row * K + o_k, mask=m_k, other=0.)
            if STRICT:
                valid = o_r < r
            else:
                valid = o_r <= r
            prod = tl.where(valid[:, None], x_r[None, :] * y_I * tl.exp(g_r[None, :] - g_I), 0.)
            tl.store(s_ptr + s_base + row * C + rows_I, tl.sum(prod, 1))

    @triton.jit
    def _scores_bwd_dx_kernel(y_ptr, g_ptr, ds_ptr, dx_ptr,
                              K: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
                              STRICT: tl.constexpr):
        # dx[i] = sum_{j<=i} dS[i, j] * y[j] * exp(gc_i - gc_j); one program per (chunk, row sub-block I).
        pid_c = tl.program_id(0)
        pid_i = tl.program_id(1)
        base = pid_c * C * K
        s_base = pid_c * C * C
        o_k = tl.arange(0, BK)
        m_k = o_k < K
        o_r = tl.arange(0, BC)
        rows_I = pid_i * BC + o_r

        g_I = tl.load(g_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        g_n = tl.load(g_ptr + base + (pid_i * BC) * K + o_k, mask=m_k, other=0.)
        acc = tl.zeros((BC, BK), dtype=tl.float32)
        for J in range(0, pid_i):
            rows_J = J * BC + o_r
            y_J = tl.load(y_ptr + base + rows_J[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
            g_J = tl.load(g_ptr + base + rows_J[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
            yd = y_J * tl.exp(g_n[None, :] - g_J)
            ds = tl.load(ds_ptr + s_base + rows_I[:, None] * C + rows_J[None, :])          # (BC, BC)
            acc += tl.dot(ds, yd, allow_tf32=False)
        acc = acc * tl.exp(g_I - g_n[None, :])

        y_I = tl.load(y_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        for r in range(0, BC):
            row = pid_i * BC + r
            g_r = tl.load(g_ptr + base + row * K + o_k, mask=m_k, other=0.)
            ds_r = tl.load(ds_ptr + s_base + row * C + rows_I)                             # (BC,)
            if STRICT:
                valid = o_r < r
            else:
                valid = o_r <= r
            prod = tl.where(valid[:, None], ds_r[:, None] * y_I * tl.exp(g_r[None, :] - g_I), 0.)
            dx_r = tl.sum(prod, 0) + tl.sum(tl.where((o_r == r)[:, None], acc, 0.), 0)
            tl.store(dx_ptr + base + row * K + o_k, dx_r, mask=m_k)

    @triton.jit
    def _scores_bwd_dy_kernel(x_ptr, g_ptr, ds_ptr, dy_ptr,
                              K: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr,
                              STRICT: tl.constexpr):
        # dy[j] = sum_{i>=j} dS[i, j] * x[i] * exp(gc_i - gc_j); one program per (chunk, column sub-block J).
        pid_c = tl.program_id(0)
        pid_j = tl.program_id(1)
        n_blocks = C // BC
        base = pid_c * C * K
        s_base = pid_c * C * C
        o_k = tl.arange(0, BK)
        m_k = o_k < K
        o_r = tl.arange(0, BC)
        rows_J = pid_j * BC + o_r

        g_J = tl.load(g_ptr + base + rows_J[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        # anchor = last row of block J (all j in J are <= it, all i in blocks I > J are >= it)
        g_n = tl.load(g_ptr + base + (pid_j * BC + BC - 1) * K + o_k, mask=m_k, other=0.)
        acc = tl.zeros((BC, BK), dtype=tl.float32)
        for I in range(pid_j + 1, n_blocks):
            rows_I = I * BC + o_r
            x_I = tl.load(x_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
            g_I = tl.load(g_ptr + base + rows_I[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
            xd = x_I * tl.exp(g_I - g_n[None, :])
            ds = tl.load(ds_ptr + s_base + rows_I[:, None] * C + rows_J[None, :])          # (BC, BC) = dS[I, J]
            acc += tl.dot(tl.trans(ds), xd, allow_tf32=False)
        acc = acc * tl.exp(g_n[None, :] - g_J)

        x_J = tl.load(x_ptr + base + rows_J[:, None] * K + o_k[None, :], mask=m_k[None, :], other=0.)
        for r in range(0, BC):
            col = pid_j * BC + r
            g_c = tl.load(g_ptr + base + col * K + o_k, mask=m_k, other=0.)
            ds_c = tl.load(ds_ptr + s_base + rows_J * C + col)                              # (BC,) = dS[I=J rows, col]
            if STRICT:
                valid = o_r > r
            else:
                valid = o_r >= r
            prod = tl.where(valid[:, None], ds_c[:, None] * x_J * tl.exp(g_J - g_c[None, :]), 0.)
            dy_c = tl.sum(prod, 0) + tl.sum(tl.where((o_r == r)[:, None], acc, 0.), 0)
            tl.store(dy_ptr + base + col * K + o_k, dy_c, mask=m_k)

    class _IntraChunkScores(torch.autograd.Function):
        BC = 16

        @staticmethod
        def forward(ctx, x, y, gc, strict):
            N, C, K = x.shape
            BC = _IntraChunkScores.BC
            assert C % BC == 0, f"chunk_size must be a multiple of {BC} for the Triton path"
            x, y, gc = x.contiguous(), y.contiguous(), gc.contiguous()
            BK = max(16, triton.next_power_of_2(K))
            s = torch.zeros((N, C, C), dtype=torch.float32, device=x.device)
            _scores_fwd_kernel[(N, C // BC)](x, y, gc, s, K=K, C=C, BC=BC, BK=BK, STRICT=strict, num_warps=4)
            ctx.save_for_backward(x, y, gc)
            ctx.strict = strict
            return s

        @staticmethod
        def backward(ctx, ds):
            x, y, gc = ctx.saved_tensors
            N, C, K = x.shape
            BC = _IntraChunkScores.BC
            BK = max(16, triton.next_power_of_2(K))
            ds = ds.contiguous()
            dx = torch.empty_like(x)
            dy = torch.empty_like(y)
            grid = (N, C // BC)
            _scores_bwd_dx_kernel[grid](y, gc, ds, dx, K=K, C=C, BC=BC, BK=BK, STRICT=ctx.strict, num_warps=4)
            _scores_bwd_dy_kernel[grid](x, gc, ds, dy, K=K, C=C, BC=BC, BK=BK, STRICT=ctx.strict, num_warps=4)
            # dS/dgc_i = x_i * (row term), dS/dgc_j = -y_j * (column term)
            dgc = x * dx - y * dy
            return dx, dy, dgc, None

    def intra_chunk_scores(x: Tensor, y: Tensor, gc: Tensor, strict: bool) -> Tensor:
        return _IntraChunkScores.apply(x, y, gc, strict)

else:  # pragma: no cover
    intra_chunk_scores = None


if __name__ == "__main__":
    # Check the kernels against the PyTorch reference (values and gradients).
    torch.manual_seed(0)
    dev = "cuda"
    for (N, C, K) in [(6, 64, 32), (5, 32, 48), (3, 64, 128), (4, 16, 16)]:
        for strict in (False, True):
            x = torch.randn(N, C, K, device=dev)
            y = torch.randn(N, C, K, device=dev)
            gc = (-torch.rand(N, C, K, device=dev) * 6).cumsum(1)   # strong decays: raw exp would overflow
            grads = {}
            outs = {}
            ds = torch.randn(N, C, C, device=dev)
            for name, fn in (("torch", intra_chunk_scores_torch), ("triton", intra_chunk_scores)):
                args = [t.clone().requires_grad_(True) for t in (x, y, gc)]
                s = fn(*args, strict)
                (s * ds).sum().backward()
                outs[name], grads[name] = s, [a.grad for a in args]
            rel = lambda a, b: ((a - b).abs().max() / a.abs().max().clamp_min(1e-30)).item()
            print(f"N={N} C={C} K={K} strict={strict} | S rel err {rel(outs['torch'], outs['triton']):.1e} | "
                  f"grad rel err " + ", ".join(f"{rel(a, b):.1e}" for a, b in zip(grads["torch"], grads["triton"])))
