"""
Sample from a trained model. Run from the project root:

$ python src/models/gpt/sample.py --out_dir=out-shakespeare-char --start="ROMEO:"
$ python src/models/gpt/sample.py --out_dir=out-shakespeare-bpe --tokenizer=gpt2bpe
$ python src/models/gpt/sample.py --init_from=gpt2 --start="Hello, I'm a language model,"
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.append(_project_root)
from src.models.gpt.model import GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-char' # relative to the current directory; ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# 'auto': char-level codec from the dataset's meta.pkl when it has one, otherwise tiktoken's gpt2 encoding.
# 'gpt2bpe': this repo's own GPT2BPETokenizer (src/models/gpt/tokenizer.py), for checkpoints trained on shakespeare_bpe.
tokenizer = 'auto'
exec(open(os.path.join(_project_root, 'configurator.py')).read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
checkpoint = None
if init_from == 'resume':
    model, checkpoint = GPT.from_checkpoint(os.path.join(out_dir, 'ckpt.pt'), device, dropout_p=0.0)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dropout_p=0.0)
else:
    raise ValueError(f"init_from must be 'resume' or a gpt2 variant, got {init_from!r}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# tokenizer
if tokenizer == 'gpt2bpe':
    from src.models.gpt.tokenizer import GPT2BPETokenizer
    tok = GPT2BPETokenizer()
    encode, decode = tok.encode, tok.decode
elif tokenizer == 'auto':
    # look for the meta pickle in case it is available in the dataset folder
    meta = None
    if checkpoint is not None and 'dataset' in checkpoint.get('config', {}): # older checkpoints might not have these...
        meta_path = os.path.join(_project_root, 'data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            print(f"Loading meta from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
    if meta is not None and 'stoi' in meta:
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No char-level meta.pkl found, assuming GPT-2 encodings...")
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
else:
    raise ValueError(f"tokenizer must be 'auto' or 'gpt2bpe', got {tokenizer!r}")

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate_sample(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
