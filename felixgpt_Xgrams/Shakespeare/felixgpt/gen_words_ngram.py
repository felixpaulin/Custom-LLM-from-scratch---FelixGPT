import word_tokens as tokens
import torch.nn.functional as F
import ngram_word_model as ngram
import re

og_input_words = input("Enter your prompt: ")
input_words = re.findall(r"\n|\w+(?:['’]\w+)?|[^\w\s]", og_input_words)
input_ids = tokens.encode(input_words)

# Add newline tokens until the context window is full
while len(input_ids) < tokens.block_size:
    input_ids.insert(0, tokens.stoi["\n"])

# Keep only the latest tokens if the prompt is too long
input_ids = input_ids[-tokens.block_size:]
generated_ids = input_ids.copy()
for i in range(1000):
    context = generated_ids[-tokens.block_size:]
    output_id = ngram.predict(context)
    generated_ids.append(output_id)

print("Generated text:", tokens.decode(generated_ids))