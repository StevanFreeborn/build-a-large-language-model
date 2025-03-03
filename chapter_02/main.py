"""Prints out file content and total number of characters in the file."""

import os
import re
from simple_tokenizer_v1 import SimpleTokenizerV1
from simple_tokenizer_v2 import SimpleTokenizerV2

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, "the_verdict.txt")

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab_size = len(all_tokens)

vocab = {token:integer for integer,token in enumerate(all_tokens)}

tokenizer = SimpleTokenizerV1(vocab)
TEXT = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(TEXT)

FINAL_TEXT_PART_ONE = "Hello, do you like tea?"
FINAL_TEXT_PART_TWO = "In the sunlit terraces of the palace."
FINAL_TEXT = " <|endoftext|> ".join((FINAL_TEXT_PART_ONE, FINAL_TEXT_PART_TWO))
print(FINAL_TEXT)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(FINAL_TEXT))
