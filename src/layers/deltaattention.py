"""Kimi Delta Attention (KDA).

KDA is the linear-attention layer from Kimi Linear (Moonshot AI, 2025): a Gated DeltaNet whose
forget gate is per-channel (one decay per key dimension) instead of a single scalar per head.

Per head, with a matrix-valued state S_t in R^{d_k x d_v}:

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

    alpha_t = exp(-exp(A) * softplus(W_alpha x_t + b_alpha))   in (0, 1)^{d_k}   (per-channel decay)
    beta_t  = sigmoid(w_beta x_t)                              in (0, 1)         (per-head write strength)

i.e. decay the state, then apply one step of the delta rule: S <- S + beta k (v - S^T k)^T.

References:
    [1] Kimi Team. Kimi Linear: An Expressive, Efficient Attention Architecture. arXiv:2510.26692, 2025.
    [2] Yang, Kautz, Hatamizadeh. Gated Delta Networks. arXiv:2412.06464, 2024.
    [3] Yang et al. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. arXiv:2406.06484, 2024.
"""

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from .normalization import RMSNorm, l2norm
from .deltaattention_triton import HAS_TRITON, intra_chunk_scores, intra_chunk_scores_torch


class GatedDeltaRule(nn.Module):
    """Gated delta rule over (batch, heads, seqlen, dim) tensors. Analogue of `DotProductAttention` for KDA.

    Inputs: query/key (b h l d_k), value (b h l d_v), log_decay (b h l d_k) with entries <= 0
    (log alpha_t), beta (b h l) in (0, 1), and an optional initial state (b h d_k d_v).
    Returns (out (b h l d_v), final_state (b h d_k d_v)). Always causal; query is scaled by d_k ** -0.5.

    use_chunk=True runs the chunkwise-parallel algorithm (sequential only over chunks); otherwise a
    token-by-token recurrence is used. Both paths compute in float32 and are numerically equivalent.
    With use_triton=True (default, CUDA only) the intra-chunk scores are computed by the fused Triton
    kernels in `deltaattention_triton.py`; the PyTorch fallback materialises a
    (b, h, seqlen, chunk_size, d_k) tensor of pairwise decays, so its memory grows with chunk_size.
    """

    def __init__(self, use_chunk: bool=True, chunk_size: int=64, use_triton: bool=True):
        super().__init__()
        self.use_chunk = use_chunk
        self.chunk_size = chunk_size
        self.use_triton = use_triton and HAS_TRITON

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                log_decay: Tensor,
                beta: Tensor,
                state: Tensor=None) -> Tuple[Tensor, Tensor]:
        out_dtype = value.dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            query, key, value, log_decay, beta = map(lambda t: t.float(), (query, key, value, log_decay, beta))
            query = query * query.shape[-1] ** -0.5
            if state is not None:
                state = state.float()

            if self.use_chunk and query.shape[2] > 1:
                out, state = self._chunk(query, key, value, log_decay, beta, state)
            else:
                out, state = self._recurrent(query, key, value, log_decay, beta, state)
        return out.to(out_dtype), state

    @staticmethod
    def _recurrent(q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor, state: Tensor):
        """Token-by-token reference recurrence."""
        b, h, l, d_k = q.shape
        if state is None:
            state = q.new_zeros(b, h, d_k, v.shape[-1])

        outs = []
        for t in range(l):
            state = state * log_decay[:, :, t].exp().unsqueeze(-1)                       # Diag(alpha_t) S
            err = v[:, :, t] - torch.einsum("b h k v, b h k -> b h v", state, k[:, :, t])  # v_t - S^T k_t
            state = state + beta[:, :, t, None, None] * k[:, :, t, :, None] * err[:, :, None, :]
            outs.append(torch.einsum("b h k v, b h k -> b h v", state, q[:, :, t]))
        return torch.stack(outs, dim=2), state

    def _chunk(self, q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor, beta: Tensor, state: Tensor):
        """Chunkwise-parallel form (WY representation, see [2, 3]) generalised to per-channel decay.

        Within a chunk let G_i = prod_{j<=i} Diag(alpha_j) (cumulative decay from the chunk start) and
        T_i = G_i^{-1} S_i. Then T_i = T_{i-1} + (G_i^{-1} k_i) u_i^T with
            u_i = beta_i (v_i - S_0^T (G_i k_i)) - sum_{j<i} A_ij u_j,   A_ij = beta_i k_i^T (G_i / G_j) k_j,
        so u = (I + A)^{-1} (beta v - beta Kbar S_0), and the outputs / chunk-final state are
            o_i = S_0^T (G_i q_i) + sum_{j<=i} [q_i^T (G_i / G_j) k_j] u_j
            S_C = G_C S_0 + sum_j ((G_C / G_j) k_j) u_j^T.
        The parts that do not depend on S_0 are computed for all chunks at once; only the state
        propagation is sequential over chunks.
        """
        b, h, l, d_k = q.shape
        d_v = v.shape[-1]
        C = self.chunk_size

        pad = (-l) % C
        if pad:
            # Padding tokens have k = 0, beta = 0 and log_decay = 0: they leave the state untouched.
            q, k, v, log_decay = map(lambda t: F.pad(t, (0, 0, 0, pad)), (q, k, v, log_decay))
            beta = F.pad(beta, (0, pad))

        q, k, v, g = map(lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=C), (q, k, v, log_decay))
        beta = rearrange(beta, "b h (n c) -> b h n c", c=C)
        n_chunks = q.shape[2]

        g = g.cumsum(dim=-2)  # log G_i: cumulative log-decay from the start of the chunk (<= 0)

        # Intra-chunk decayed scores S[i, j] = x_i^T (G_i / G_j) y_j, lower-triangular:
        #   P (j <= i): q against k      A (j < i): beta k against k
        scores = intra_chunk_scores if (self.use_triton and q.is_cuda) else intra_chunk_scores_torch
        flat = lambda t: rearrange(t, "b h n c d -> (b h n) c d")
        k_flat, g_flat = flat(k), flat(g)
        P = rearrange(scores(flat(q), k_flat, g_flat, False), "(b h n) i j -> b h n i j", b=b, h=h)
        A = rearrange(scores(flat(k * beta[..., None]), k_flat, g_flat, True), "(b h n) i j -> b h n i j", b=b, h=h)

        # (I + A) is unit lower-triangular: solve for u_tilde = (I+A)^{-1} (beta v) and W = (I+A)^{-1} (beta Kbar)
        I_plus_A = A + torch.eye(C, dtype=A.dtype, device=A.device)
        k_bar = k * g.exp()                                                          # G_i k_i
        q_bar = q * g.exp()                                                          # G_i q_i
        rhs = torch.cat([v, k_bar], dim=-1) * beta[..., None]
        sol = torch.linalg.solve_triangular(I_plus_A, rhs, upper=False, unitriangular=True)
        u_tilde, W = sol.split([d_v, d_k], dim=-1)

        k_end = k * (g[..., -1:, :] - g).exp()                                       # (G_C / G_j) k_j
        decay_end = g[..., -1, :].exp()                                              # G_C

        if state is None:
            state = q.new_zeros(b, h, d_k, d_v)

        # unbind once: slicing inside the loop makes autograd allocate a full-size zero tensor per slice
        u_tilde, W, q_bar, P, k_end_T, decay_end = (
            t.unbind(dim=2) for t in (u_tilde, W, q_bar, P, k_end.transpose(-1, -2), decay_end.unsqueeze(-1)))
        outs = []
        for i in range(n_chunks):
            u = u_tilde[i] - W[i] @ state
            outs.append(q_bar[i] @ state + P[i] @ u)
            state = decay_end[i] * state + k_end_T[i] @ u

        out = rearrange(torch.stack(outs, dim=2), "b h n c d -> b h (n c) d")[:, :, :l]
        return out, state


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention layer: drop-in analogue of `GroupedQueryRotaryAttention` with a recurrent state.

    x (b l embed_dim) -> q, k, v projections -> depthwise causal short conv + SiLU -> L2-normalised q, k
    -> gated delta rule -> per-head RMSNorm gated by SiLU(W_g x) -> output projection.

    Always causal, so no attention mask is needed; `attn_mask` is accepted for interface compatibility
    and must be causal (nothing masked on or below the diagonal). No positional embedding is needed:
    the recurrence and the short conv are position-aware.

    Instead of a KV cache the layer keeps `state_cache` = (recurrent_state, conv_state_q, conv_state_k,
    conv_state_v, seqlen) when `use_kv_cache=True`, with the same get/clear methods as the other layers.
    Unlike a KV cache the recurrent state cannot be trimmed to an earlier position.
    """

    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 dim_head: int=None,
                 expand_v: float=1.,
                 conv_size: int=4,
                 use_short_conv: bool=True,
                 chunk_size: int=64,
                 use_chunk: bool=True,
                 use_triton: bool=True,
                 bias: bool=False,
                 norm_eps: float=1e-5,
                 batch_first: bool=True):
        super().__init__()
        if dim_head is None:
            assert embed_dim % n_heads == 0
            dim_head = embed_dim // n_heads

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dim_head = dim_head
        self.dim_head_v = int(dim_head * expand_v)
        self.key_dim = n_heads * dim_head
        self.value_dim = n_heads * self.dim_head_v
        self.conv_size = conv_size
        self.use_short_conv = use_short_conv
        self.state_cache: Tuple[Tensor, Tensor, Tensor, Tensor, int] = None

        self.delta_rule = GatedDeltaRule(use_chunk=use_chunk, chunk_size=chunk_size, use_triton=use_triton)
        self.to_q = nn.Linear(embed_dim, self.key_dim, bias=bias)
        self.to_k = nn.Linear(embed_dim, self.key_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, self.value_dim, bias=bias)
        self.to_out = nn.Linear(self.value_dim, embed_dim, bias=bias)

        if use_short_conv:
            # Depthwise causal conv (Mamba-style); inputs are left-padded / cached by conv_size - 1 in forward.
            self.q_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, bias=False)
            self.k_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, bias=False)
            self.v_conv = nn.Conv1d(self.value_dim, self.value_dim, conv_size, groups=self.value_dim, bias=False)

        # Per-channel forget gate alpha_t = exp(-exp(A_log) * softplus(W x_t + dt_bias)); low-rank W.
        self.to_decay = nn.Sequential(nn.Linear(embed_dim, dim_head, bias=False),
                                      nn.Linear(dim_head, self.key_dim, bias=False))
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(1, 16).log())
        self.dt_bias = nn.Parameter(self._init_dt_bias(self.key_dim))
        # Per-head write strength beta_t = sigmoid(w x_t).
        self.to_beta = nn.Linear(embed_dim, n_heads, bias=False)

        # Output gate (low-rank) and per-head gated RMSNorm.
        self.to_gate = nn.Sequential(nn.Linear(embed_dim, dim_head, bias=False),
                                     nn.Linear(dim_head, self.value_dim, bias=False))
        self.o_norm = RMSNorm(self.dim_head_v, eps=norm_eps)

    @staticmethod
    def _init_dt_bias(dim: int, dt_min: float=1e-3, dt_max: float=1e-1, dt_init_floor: float=1e-4) -> Tensor:
        """Mamba-style init so that softplus(dt_bias) is log-uniform in [dt_min, dt_max]."""
        dt = torch.exp(torch.rand(dim) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        return dt + torch.log(-torch.expm1(-dt))  # inverse softplus

    def forward(self,
                x: Tensor,
                attn_mask: Tensor=None,
                use_kv_cache: bool=False):
        if attn_mask is not None and attn_mask.bool().tril().any():
            raise ValueError("KimiDeltaAttention is always causal; attn_mask may only mask future positions.")

        if use_kv_cache:
            if self.training:
                raise RuntimeError("KimiDeltaAttention must be in .eval() mode if using KV caching")
            if self.state_cache is not None:
                state, conv_q, conv_k, conv_v, seqlen = self.state_cache
            else:
                state = conv_q = conv_k = conv_v = None
                seqlen = 0
        else:
            state = conv_q = conv_k = conv_v = None

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        if self.use_short_conv:
            q, conv_q = self._causal_conv(self.q_conv, q, conv_q)
            k, conv_k = self._causal_conv(self.k_conv, k, conv_k)
            v, conv_v = self._causal_conv(self.v_conv, v, conv_v)

        q = l2norm(rearrange(q, "b l (h d) -> b h l d", h=self.n_heads))
        k = l2norm(rearrange(k, "b l (h d) -> b h l d", h=self.n_heads))
        v = rearrange(v, "b l (h d) -> b h l d", h=self.n_heads)

        # log alpha_t <= 0, shape (b h l dim_head); beta_t in (0, 1), shape (b h l)
        decay_logits = self.to_decay(x).float() + self.dt_bias
        decay_logits = rearrange(decay_logits, "b l (h d) -> b h l d", h=self.n_heads)
        log_decay = -self.A_log.float().exp()[None, :, None, None] * F.softplus(decay_logits)
        beta = rearrange(self.to_beta(x).sigmoid(), "b l h -> b h l")

        attn_output, state = self.delta_rule(q, k, v, log_decay, beta, state)

        if use_kv_cache:
            self.state_cache = (state.detach(),
                                None if conv_q is None else conv_q.detach(),
                                None if conv_k is None else conv_k.detach(),
                                None if conv_v is None else conv_v.detach(),
                                seqlen + x.shape[1])

        gate = rearrange(self.to_gate(x), "b l (h d) -> b h l d", h=self.n_heads)
        attn_output = self.o_norm(attn_output) * F.silu(gate)
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
        out = self.to_out(attn_output)
        return out

    def _causal_conv(self, conv: nn.Conv1d, x: Tensor, conv_state: Tensor=None) -> Tuple[Tensor, Tensor]:
        """Depthwise causal conv + SiLU on x (b l d). conv_state holds the previous conv_size - 1 inputs."""
        x = rearrange(x, "b l d -> b d l")
        if conv_state is None:
            conv_state = x.new_zeros(x.shape[0], x.shape[1], self.conv_size - 1)
        x = torch.cat([conv_state, x], dim=-1)
        new_conv_state = x[:, :, -(self.conv_size - 1):]
        x = F.silu(conv(x))
        return rearrange(x, "b d l -> b l d"), new_conv_state

    def get_kv_cache_seqlen(self):
        """Number of tokens absorbed into the cached recurrent state."""
        if self.state_cache is None:
            return 0
        else:
            return self.state_cache[-1]

    def clear_kv_cache(self):
        """Clears the cached recurrent / conv state."""
        self.state_cache = None

    def __repr__(self):
        return (f'KimiDeltaAttention(embed_dim={self.embed_dim}, n_heads={self.n_heads}, '
                f'dim_head={self.dim_head}, dim_head_v={self.dim_head_v}, '
                f'use_short_conv={self.use_short_conv}, chunk_size={self.delta_rule.chunk_size})')


if __name__ == "__main__":
    # Self-checks: chunk vs recurrent equivalence (values and grads), and cached incremental decoding.
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, h, l, d_k, d_v = 2, 3, 100, 32, 48

    base_inputs = (l2norm(torch.randn(b, h, l, d_k, device=device)),   # q, k unit-norm as in the layer
                   l2norm(torch.randn(b, h, l, d_k, device=device)),
                   torch.randn(b, h, l, d_v, device=device),
                   # strong decays: exp of the raw cumulative sums would overflow without masking
                   -torch.rand(b, h, l, d_k, device=device) * 8,
                   torch.rand(b, h, l, device=device))
    state0 = torch.randn(b, h, d_k, d_v, device=device)
    results = {}
    for name, op in (("recurrent", GatedDeltaRule(use_chunk=False)),
                     ("chunk-torch", GatedDeltaRule(use_chunk=True, chunk_size=32, use_triton=False)),
                     ("chunk-triton", GatedDeltaRule(use_chunk=True, chunk_size=32, use_triton=True))):
        inputs = [t.clone().requires_grad_(True) for t in base_inputs]
        out, final_state = op(*inputs, state=state0)
        (out.square().sum() + final_state.square().sum()).backward()
        results[name] = (out, final_state, [t.grad for t in inputs])

    rel = lambda a, c: ((a - c).abs().max() / a.abs().max()).item()
    out_r, state_r, grads_r = results["recurrent"]
    for name in ("chunk-torch", "chunk-triton"):
        out_c, state_c, grads_c = results[name]
        print(f"{name} vs recurrent | out rel err {rel(out_r, out_c):.2e} "
              f"| state rel err {rel(state_r, state_c):.2e} "
              f"| grad rel err {max(rel(a, c) for a, c in zip(grads_r, grads_c)):.2e}")

    layer = KimiDeltaAttention(embed_dim=64, n_heads=4, chunk_size=16).to(device).eval()
    x = torch.randn(b, l, 64, device=device)
    with torch.no_grad():
        full = layer(x)
        layer.clear_kv_cache()
        prefill = 37
        parts = [layer(x[:, :prefill], use_kv_cache=True)]
        for t in range(prefill, l):
            parts.append(layer(x[:, t:t + 1], use_kv_cache=True))
        incremental = torch.cat(parts, dim=1)
    print(f"full vs incremental (cached) | max err {(full - incremental).abs().max():.2e} "
          f"| cache seqlen {layer.get_kv_cache_seqlen()} (expected {l})")
    print(layer)
