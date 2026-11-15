import step1_tokens as tokens
import torch.nn.functional as F
import bigram

input_char = input("Enter a character: ")

generated_text = f"{input_char}"

input_id = tokens.stoi[input_char]

for i in range(200):

    output_char, output_id = bigram.predict(input_id)
    input_id = output_id
    generated_text += output_char

print("Generated text:", generated_text)


"""
bigram.predictTop5(input_id)

print("Top 5 predictions:", generated_text)

"""
    