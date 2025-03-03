"""Simple tokenizer that splits text by spaces."""

import re

class SimpleTokenizerV1:
    """Tokenizes text by splitting it by spaces."""
    def __init__(self, v):
        self.str_to_int = v
        self.int_to_str = {i:s for s,i in v.items()}

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
        text = " ".join([self.int_to_str[i] for i in identifiers])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
