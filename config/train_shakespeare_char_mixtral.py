# train a miniature character-level Mixtral (mixture of experts) on shakespeare
# 6 layers, 384 dim, 6 heads / 2 kv heads, 4 experts of SwiGLU width 512 with top-2 routing;
# ~16.6M parameters total, ~9M active per token

out_dir = 'out-shakespeare-char-mixtral'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-mixtral'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'mixtral'
n_layers = 6
n_heads = 6
n_kv_heads = 2
embed_dim = 384
ffn_hidden_dim = 512
n_experts = 4
n_experts_per_tok = 2
dropout_p = 0.0 # Mixtral has no dropout (best val ~1.51 at step 500, then it memorises)

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

compile = False # the MoE dispatch uses data-dependent masks, which torch.compile keeps recompiling
