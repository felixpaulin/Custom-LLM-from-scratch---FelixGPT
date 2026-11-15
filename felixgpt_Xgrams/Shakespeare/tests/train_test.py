import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml

vocab, dim = 1000, 512

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)   # token id -> vector
        self.ln = nn.LayerNorm(dim)           # keeps numbers stable
        self.fc = nn.Linear(dim, vocab)       # vector -> score for each token

    def forward(self, x):
        return self.fc(self.ln(self.emb(x)))

def run(device, steps=200):
    torch.manual_seed(0)
    model = Tiny().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, vocab, (64, 64)).to(device)
    y = ((x + 1) % vocab)                     # simple rule to learn: next = id + 1
    start = time.time()
    for step in range(steps):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        opt.zero_grad()
        loss.backward()                       # backpropagation
        opt.step()                            # update the weights
        if step == 0:
            first = loss.item()
    last = loss.item()                        # .item() also waits for the GPU
    return first, last, time.time() - start

dml = torch_directml.device()
for name, dev in [("CPU", torch.device("cpu")), ("GPU", dml)]:
    first, last, t = run(dev)
    print(f"{name}: loss {first:.2f} -> {last:.2f} in {t:.1f}s")