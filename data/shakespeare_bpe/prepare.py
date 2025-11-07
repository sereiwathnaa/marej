"""
Prepare the Shakespeare dataset for BPE tokenization using GPT2BPETokenizer.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
tokenizer vocab size.
"""
import os
import sys
import pickle
import requests
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.models.gpt.tokenizer import GPT2BPETokenizer

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Initialize GPT2 BPE tokenizer
print("Initializing GPT2BPETokenizer...")
tokenizer = GPT2BPETokenizer()
vocab_size = len(tokenizer.token_to_index)
print(f"vocab size: {vocab_size:,}")

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers using BPE
print("Encoding training data...")
train_ids = tokenizer.encode(train_data)
print("Encoding validation data...")
val_ids = tokenizer.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'tokenizer_type': 'gpt2_bpe',
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done!")
