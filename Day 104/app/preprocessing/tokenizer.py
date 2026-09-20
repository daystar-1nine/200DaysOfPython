"""
Tokenizer module for Day 104 Semantic Search Engine.
Extracts words, sub-tokens, and n-grams from cleaned text.
"""

import re
from typing import List


class Tokenizer:
    """Tokenizes text into word tokens and n-grams."""

    def __init__(self, min_token_len: int = 1):
        self.min_token_len = min_token_len

    def tokenize(self, text: str) -> List[str]:
        """Split text into word tokens using word boundary matching.
        
        Args:
            text: Input text string.
            
        Returns:
            List of string tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        tokens = re.findall(r"\b\w+\b", text)
        if self.min_token_len > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_len]
        return tokens

    @staticmethod
    def get_ngrams(tokens: List[str], n: int = 2) -> List[str]:
        """Extract n-grams from a list of tokens.
        
        Args:
            tokens: List of tokens.
            n: N-gram order.
            
        Returns:
            List of joined n-gram strings.
        """
        if n <= 0 or len(tokens) < n:
            return []
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
