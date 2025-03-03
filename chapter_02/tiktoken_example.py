"""A tokenization example using the TikToken lib"""

import os
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, "the_verdict.txt")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

CONTEXT_SIZE = 4
x = enc_sample[:CONTEXT_SIZE]
y = enc_sample[1:CONTEXT_SIZE+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, CONTEXT_SIZE+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

for i in range(1, CONTEXT_SIZE+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
