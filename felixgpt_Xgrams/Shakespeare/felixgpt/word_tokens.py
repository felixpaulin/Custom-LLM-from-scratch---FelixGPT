import regex as re
import torch

block_size = 8

with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Original word-level regex (V1–V14)
# pattern = r"\r\n|\n|\w+(?:['’]\w+)?|[^\w\s]"

words = re.findall(pattern, text)

vocab_size = len(set(words))
vocab = sorted(set(words))

print("Vocabulary size:", vocab_size)
print("Total tokens:", len(words))
print("Newline tokens:", words.count("\n"))
print("Carriage return tokens:", words.count("\r"))
print("Total tokens:", len(words))

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}

def encode(s):
    return [stoi[w] for w in s]


def decode(ids):
    decoded = ""
    for token_id in ids:
        token = itos[int(token_id)]
        decoded += token
    return decoded
"""
# old decoder for V1-14
def decode(ids):
    decoded = ""
    for token_id in ids:
        token = itos[int(token_id)]

        if token == "\n":
            decoded += "\n"
        elif token in [".", ",", "!", "?", ";", ":", "'", ")", "]"]:
            decoded += token
        elif token in ["(", "["]:
            decoded += " " + token
        else:
            decoded += " " + token

    return decoded
"""
n = int(0.9 * len(words))
train_data = torch.tensor(encode(words[:n]), dtype=torch.long)
val_data = torch.tensor(encode(words[n:]), dtype=torch.long)

torch.manual_seed(1337)
batch_size = 32

def get_ngram_batch(split, block_size):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + block_size] for i in ix])
    return x, y

test = "I say unto you, what he hath done famously, he did it to that end:"

print(decode(encode(re.findall(pattern, test))))