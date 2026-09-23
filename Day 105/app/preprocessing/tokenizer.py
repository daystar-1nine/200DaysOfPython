"""
Tokenizer module for Day 105: Neural NLP & Text Classification.
"""

import re
from typing import List


class Tokenizer:
    """Tokenizes raw text into normalized word tokens."""

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        """Convert string to list of tokens using regex word boundaries.
        
        Args:
            text: Input string.
            
        Returns:
            List of string tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        if self.lowercase:
            text = text.lower()

        # Extract words and handle alphanumeric tokens
        tokens = re.findall(r"\b\w+\b", text)
        return tokens
