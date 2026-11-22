# Use this command in PS to navigate to felixgpt from the root folder quickly:
# "cd felixgpt_Xgrams && cd Shakespeare && cd felixgpt"

import torch
import word_tokens as tokens
import torch.nn.functional as F
import torch_directml
import time

start_time = time.time()

device = torch_directml.device()
print("Using device:", device)

temperature = 0.8
embedding_dim = 72
hidden_dim = 136
hidden_dim2 = 192
hidden_dim3 = 224
batch_size = 32

version = "V2.31"

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
        self.hidden = torch.nn.Linear(embedding_dim * block_size, hidden_dim)
        # remove this line for model before V2.25
        self.hidden2 = torch.nn.Linear(hidden_dim, hidden_dim2)
        # remove this line for models before V2.28
        self.hidden3 = torch.nn.Linear(hidden_dim2, hidden_dim3)
        # use this for models before V2.21 insted of the three lines: self.output = torch.nn.Linear(hidden_dim, vocab_size)
        self.output_projection = torch.nn.Linear(hidden_dim3, embedding_dim)
        self.output = torch.nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output.weight = self.embedding_table.weight

        self.dropout = torch.nn.Dropout(0.35)

    def forward(self, idx):
        x = self.embedding_table(idx)
        x = x.flatten(start_dim=1)
        x = self.hidden(x)
        # use F.relu(x) instead of gelu for versions before V2.30
        # use F.gelu(x) instead of silu or relu for V2.30
        x = F.silu(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.hidden3(x)
        x= F.silu(x)
        x= self.dropout(x)
        x = self.output_projection(x)

        logits = self.output(x)
        return logits

model = NGramLanguageModel(tokens.vocab_size, embedding_dim, tokens.block_size).to(device)
param_count = sum(p.numel() for p in model.parameters())

model.load_state_dict(torch.load(f"../checkpoints/felixgpt_word_{version}_best.pth", map_location=device)) 
"""
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000, eta_min=0.00005)

xb, yb = tokens.get_ngram_batch("train", tokens.block_size)
xb = xb.to(device) 
yb = yb.to(device)

logits = model(xb)
print("Input shape:", xb.shape)
print("Logits shape:", logits.shape)

training_losses = []
best_val_loss = float("inf")

print(f"Training FelixGPT Ngram model {version}!")

for iterations in range(150000):
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
print(f"Number of parameters: {param_count}")
"""
def predict(input_ids):
    model.eval()
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits / temperature, dim=-1)
        output_id = torch.multinomial(probs, num_samples=1).item()
    return output_id