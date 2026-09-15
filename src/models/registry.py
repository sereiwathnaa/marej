"""Model registry shared by train.py and sample.py: build a model from config keys, load a checkpoint,
and the training helpers (optimizer groups, MFU estimate) that used to live on the GPT class.

Every model type maps the common config keys (n_layers, n_heads, embed_dim, block_size, vocab_size, ...)
to its own constructor. Checkpoints store the full `model_args` dict (including model_type), so
`load_checkpoint` can rebuild any model without knowing its type in advance.
"""

import math

import torch


def build_model(model_type: str, **args):
    """Instantiate a model from the training config keys (unused keys are ignored per model type)."""
    if model_type == 'gpt':
        from src.models.gpt.model import GPT
        return GPT(n_layers=args['n_layers'], n_heads=args['n_heads'], embed_dim=args['embed_dim'],
                   vocab_size=args['vocab_size'], block_size=args['block_size'],
                   dropout_p=args['dropout_p'], bias=args['bias'])
    raise ValueError(f"unknown model_type {model_type!r}")


def load_checkpoint(ckpt_path: str, device: str, dropout_p: float=None):
    """Load a checkpoint written by train.py. Returns (model, checkpoint dict).

    Strips the '_orig_mod.' prefix that torch.compile adds to state dict keys. Older GPT checkpoints
    without a model_type are treated as 'gpt'.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = dict(checkpoint['model_args'])
    model_args.setdefault('model_type', 'gpt')
    if dropout_p is not None:
        model_args['dropout_p'] = dropout_p
    model = build_model(**model_args)
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    return model, checkpoint


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """AdamW with weight decay on matrices/embeddings only (biases and norm weights are not decayed)."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=(device_type == 'cuda'))


def estimate_mfu(model, fwdbwd_per_iter, dt):
    """Model flops utilization as a fraction of A100 bfloat16 peak (PaLM paper appendix B).

    Only meaningful for attention models with n_layers / n_heads / embed_dim / block_size attributes;
    returns -1 otherwise (e.g. Mamba).
    """
    if not all(hasattr(model, a) for a in ('n_layers', 'n_heads', 'embed_dim', 'block_size')):
        return -1.0
    N = sum(p.numel() for p in model.parameters())
    L, H, Q, T = model.n_layers, model.n_heads, model.embed_dim // model.n_heads, model.block_size
    flops_per_token = 6 * N + 12 * L * H * Q * T
    flops_per_iter = flops_per_token * T * fwdbwd_per_iter
    return flops_per_iter / dt / 312e12
