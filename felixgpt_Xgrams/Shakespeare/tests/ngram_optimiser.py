import random
import math
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_directml
from felixgpt import word_tokens as tokens

device = torch_directml.device()
print("Using device:", device)

vocab_size = tokens.vocab_size
block_size = tokens.block_size

# -----------------------------
# CONFIGURATION
# -----------------------------

vocab_size = 14298
block_size = 8
max_parameters = 1_500_000

total_experiments = 10

random.seed(1337)


# -----------------------------
# CONFIGURATION GENERATOR
# -----------------------------

def generate_config():

    while True:

        embedding_dim = random.choice([16, 24, 32, 40, 48])
        hidden_width = random.choice([32, 40, 48, 56, 64, 72, 80])

        dropout = random.uniform(0.1, 0.4)
        learning_rate = 10 ** random.uniform(-4, -3)
        batch_size = random.choice([16, 32, 64])

        config = {
            "embedding_dim": embedding_dim,
            "hidden_width": hidden_width,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }

        if calculate_parameters(config) <= max_parameters:
            return config

# -----------------------------
# PARAMETER CALCULATOR
# -----------------------------

def calculate_parameters(config):

    E = config["embedding_dim"]
    H = config["hidden_width"]
    V = vocab_size
    B = block_size

    embedding_params = V * E

    hidden_params = (B * E * H) + H

    output_params = (H * V) + V

    total = embedding_params + hidden_params + output_params

    return total

def generate_valid_configs():

    configs = []

    while len(configs) < total_experiments:

        config = generate_config()
        params = calculate_parameters(config)

        if params <= max_parameters:
            configs.append(config)

    return configs

def verify_parameters(model, config):

    calculated = calculate_parameters(config)
    actual = sum(p.numel() for p in model.parameters())
    assert calculated == actual, (
        f"Parameter mismatch! Calculated: {calculated}, Actual: {actual}"
    )

    return actual

class NGramLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_width, dropout, block_size):

        super().__init__()

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)

        self.hidden = nn.Linear(
            embedding_dim * block_size,
            hidden_width
        )

        self.output = nn.Linear(hidden_width, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):

        x = self.embedding_table(idx)
        x = x.flatten(start_dim=1)

        x = self.hidden(x)
        x = F.relu(x)
        x = self.dropout(x)

        logits = self.output(x)

        return logits

def train_experiment(config, iterations=2000):

    model = NGramLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config["embedding_dim"],
        hidden_width=config["hidden_width"],
        dropout=config["dropout"],
        block_size=block_size
    ).to(device)

    actual_params = verify_parameters(model, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )
    start_time = time.time()
    for i in range(iterations):
        model.train()
        xb, yb = tokens.get_ngram_batch("train", block_size)
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
    training_time = time.time() - start_time
    return model, actual_params, training_time

def evaluate_model(model, batches=100):

    model.eval()

    losses = []

    with torch.no_grad():

        for i in range(batches):

            xb, yb = tokens.get_ngram_batch("val", block_size)

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = F.cross_entropy(logits, yb)

            losses.append(loss.item())

    return sum(losses) / len(losses)

results = []

for i in range(total_experiments):

    config = generate_config()

    params = calculate_parameters(config)

    print(f"\n{'='*35}")
    print(f"EXPERIMENT {i + 1}/{total_experiments}")
    print(f"{'='*35}")

    print("Configuration:", config)
    print(f"Parameters: {params:,}")

    model, actual_params, training_time = train_experiment(config)

    val_loss = evaluate_model(model)

    result = {
        "config": config,
        "parameters": actual_params,
        "validation_loss": val_loss,
        "training_time": training_time
    }

    results.append(result)

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

results.sort(key=lambda x: x["validation_loss"])

print("\n\n========== LEADERBOARD ==========")

for rank, result in enumerate(results, 1):

    print(f"\nRank: {rank}")
    print(f"Validation loss: {result['validation_loss']:.4f}")
    print(f"Parameters: {result['parameters']:,}")
    print(f"Training time: {result['training_time']:.2f}s")
    print("Configuration:", result["config"])