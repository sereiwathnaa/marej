# train a miniature character-level Llama on shakespeare (same size/budget as train_shakespeare_char.py)
# rotary GQA attention (6 heads, 2 kv heads), SwiGLU FFN, RMSNorm; ~9.5M parameters

out_dir = 'out-shakespeare-char-llama'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-llama'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'llama'
n_layers = 6
n_heads = 6
n_kv_heads = 2
embed_dim = 384
ffn_hidden_dim = 1024
dropout_p = 0.0 # Llama has no dropout, so it overfits sooner than the GPT config (best val ~1.56 at step 500)

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
