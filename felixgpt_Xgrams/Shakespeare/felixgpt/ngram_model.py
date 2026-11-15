import torch
import step1_tokens as tokens
import torch.nn.functional as F
import torch_directml

device = torch_directml.device()
print("Using device:", device)

temperature = 0.8
embedding_dim = 32
batch_size = 16

def estimate_loss(model):
    model.eval()
    losses = 0
    for i in range(50):
        with torch.no_grad():
            xb, yb = tokens.get_ngram_batch("val", tokens.block_size)
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            losses += loss.item()
    model.train()
    return losses / 50

class NGramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=embedding_dim, block_size=tokens.block_size):
        super().__init__()

        self.embedding_table = torch.nn.Embedding(vocab_size, embedding_dim)

        self.hidden = torch.nn.Linear(embedding_dim * block_size, 128)
        self.output = torch.nn.Linear(128, vocab_size)

    def forward(self, idx):
        x = self.embedding_table(idx)
        x = x.flatten(start_dim=1)
        x = self.hidden(x)

        x = F.relu(x)
        logits = self.output(x)
        return logits

model = NGramLanguageModel(tokens.vocab_size, embedding_dim, tokens.block_size).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {param_count}")

model.load_state_dict(torch.load("../checkpoints/felixgpt_Ngram_checkpoint_85k.pth", map_location=device)) 

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
xb, yb = tokens.get_ngram_batch("train", tokens.block_size)
xb = xb.to(device) 
yb = yb.to(device)

logits = model(xb)
print("Input shape:", xb.shape)
print("Logits shape:", logits.shape)

"""
training_losses = []

for iterations in range(85000):
    optimizer.zero_grad()
    xb, yb = tokens.get_ngram_batch("train", tokens.block_size)
    xb = xb.to(device)
    yb = yb.to(device)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    if len(training_losses) == 5000:
        average_loss = sum(training_losses) / len(training_losses)
        print(f"Iteration {iterations + 1}")
        print(f"Loss (train): {average_loss}") 
        print(f"Loss (val): {estimate_loss(model)} ")
        training_losses.clear()

torch.save(model.state_dict(), "../checkpoints/felixgpt_Ngram_checkpoint_85k.pth")
print("FelixGPT Ngram checkpoint saved!")
"""

def predict(input_ids):
    model.eval()
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits / temperature, dim=-1)
        output_id = torch.multinomial(probs, num_samples=1).item()
    return output_id