"""Prints out file content and total number of characters in the file."""

import os
import re

class SimpleTokenizerV1:
    """Tokenizes text by splitting it by spaces."""
    def __init__(self, v):
        self.str_to_int = v
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text1):
        """Converts text to a list of integers."""
        pre = re.split(r'([,.?_!"()\']|--|\s)', text1)
        pre = [
         item.strip() for item in pre if item.strip()
        ]
        identifiers = [self.str_to_int[s] for s in pre]
        return identifiers

    def decode(self, identifiers):
        """Converts a list of integers to text."""
        text4 = " ".join([self.int_to_str[i] for i in identifiers])
        text4 = re.sub(r'\s+([,.?!"()\'])', r'\1', text4)
        return text4

class SimpleTokenizerV2:
    """Tokenizes text by splitting it by spaces."""
    def __init__(self, vocab2):
        self.str_to_int = vocab2
        self.int_to_str = { i:s for s,i in vocab2.items()}

    def encode(self, text2):
        """Converts text to a list of integers."""
        preprocessed2 = re.split(r'([,.:;?_!"()\']|--|\s)', text2)
        preprocessed2 = [
            item.strip() for item in preprocessed2 if item.strip()
        ]
        preprocessed2 = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed2]
        ids2 = [self.str_to_int[s] for s in preprocessed2]
        return ids2

    def decode(self, ids3):
        """Converts a list of integers to text."""
        text3 = " ".join([self.int_to_str[i] for i in ids3])
        text3 = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text3)
        return text3

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

finalTextPartOne = "Hello, do you like tea?"
finalTextPartTwo = "In the sunlit terraces of the palace."
finalText = " <|endoftext|> ".join((finalTextPartOne, finalTextPartTwo))
print(finalText)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(finalText))
