import step1_tokens as tokens
import torch.nn.functional as F
import ngram_model as ngram

input_chars = input("Enter up to 8 characters: ")
if len(input_chars) > tokens.block_size:
    raise SystemExit(f"Please enter no more than {tokens.block_size} characters.")
input_chars = input_chars.ljust(tokens.block_size, "\n")

generated_text = f"{input_chars}"

input_ids = [tokens.stoi[c] for c in input_chars]

for i in range(1000):
    output_id = ngram.predict(input_ids)
    output_char = tokens.itos[output_id]
    generated_text += output_char
    input_ids = input_ids[1:] + [output_id]

print("Generated text:", generated_text)