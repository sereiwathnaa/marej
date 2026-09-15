# train a miniature character-level Mamba on shakespeare
# 6 Mamba blocks, d_model 384 (inner 768), d_state 16; ~5.8M parameters
#
# The selective scan is pure PyTorch (chunked parallel scan + gradient checkpointing, see
# src/models/mamba/model.py), which is far slower and hungrier than the fused CUDA kernel, so the
# batch / context / iteration budget is smaller than the transformer configs.

out_dir = 'out-shakespeare-char-mamba'
eval_interval = 250
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-mamba'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

model_type = 'mamba'
n_layers = 6
embed_dim = 384
d_state = 16
expand = 2
dropout_p = 0.0 # Mamba has no dropout (best val ~1.54 at step 750 with this budget)

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

compile = False # the checkpointed scan does not benefit from torch.compile
