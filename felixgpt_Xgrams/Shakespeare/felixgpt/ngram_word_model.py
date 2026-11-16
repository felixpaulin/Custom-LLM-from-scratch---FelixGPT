import torch
import word_tokens as tokens
import torch.nn.functional as F
import torch_directml

device = torch_directml.device()
print("Using device:", device)

temperature = 0.8
embedding_dim = 64
batch_size = 32

version = "V2.18"

def estimate_loss(model):
    model.eval()
    losses = 0
    for i in range(500):
        with torch.no_grad():
            xb, yb = tokens.get_ngram_batch("val", tokens.block_size)
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            losses += loss.item()
    model.train()
    return losses / 500

class NGramLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=embedding_dim, block_size=tokens.block_size):
        super().__init__()

        self.embedding_table = torch.nn.Embedding(vocab_size, embedding_dim)

        self.hidden = torch.nn.Linear(embedding_dim * block_size, 160)
        self.output = torch.nn.Linear(160, vocab_size)
        self.dropout = torch.nn.Dropout(0.35)

    def forward(self, idx):
        x = self.embedding_table(idx)
        x = x.flatten(start_dim=1)
        x = self.hidden(x)

        x = F.relu(x)
        x = self.dropout(x)

        logits = self.output(x)
        return logits

model = NGramLanguageModel(tokens.vocab_size, embedding_dim, tokens.block_size).to(device)
param_count = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {param_count}")

model.load_state_dict(torch.load(f"../checkpoints/felixgpt_word_{version}_best.pth", map_location=device)) 
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100000, eta_min=0.00005)

xb, yb = tokens.get_ngram_batch("train", tokens.block_size)
xb = xb.to(device) 
yb = yb.to(device)

logits = model(xb)
print("Input shape:", xb.shape)
print("Logits shape:", logits.shape)

training_losses = []
best_val_loss = float("inf")

print(f"Training FelixGPT Ngram model {version}!")

for iterations in range(100000):
    optimizer.zero_grad()
    xb, yb = tokens.get_ngram_batch("train", tokens.block_size)
    xb = xb.to(device)
    yb = yb.to(device)
    logits = model(xb)
    loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
    loss.backward()
    optimizer.step()
    scheduler.step()
    training_losses.append(loss.item())

    if (iterations + 1) % 1000 == 0:
        val_loss = estimate_loss(model)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"../checkpoints/felixgpt_word_{version}_best.pth")
            print(f"New best checkpoint saved! Loss (val): {best_val_loss}, Iteration: {iterations + 1}")

    if len(training_losses) == 5000:
        average_loss = sum(training_losses) / len(training_losses)
        print(f"Iteration {iterations + 1}")
        print(f"Loss (train): {average_loss}") 
        print(f"Loss (val): {val_loss}")
        training_losses.clear()

print(f"{version} training finished!")
print(f"Best validation loss: {best_val_loss}")
"""
def predict(input_ids):
    model.eval()
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits / temperature, dim=-1)
        output_id = torch.multinomial(probs, num_samples=1).item()
    return output_id