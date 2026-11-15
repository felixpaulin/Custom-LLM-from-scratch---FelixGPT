import torch

block_size = 8

with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

def get_ngram_batch(split, n):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - n, (batch_size,))
    x = torch.stack([d[i:i + n] for i in ix])
    y = torch.stack([d[i + n] for i in ix])
    return x, y