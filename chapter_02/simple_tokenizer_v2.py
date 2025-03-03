"""Simple tokenizer class."""

import re

class SimpleTokenizerV2:
    """Tokenizes text by splitting it by spaces."""
    def __init__(self, v):
        self.str_to_int = v
        self.int_to_str = { i:s for s,i in v.items()}

    def encode(self, text):
        """Converts text to a list of integers."""
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        """Converts a list of integers to text."""
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
