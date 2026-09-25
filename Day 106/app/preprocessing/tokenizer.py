"""
Word tokenizer module for Day 106: RNNs & Sequential Text Learning.
Splits text into discrete word and punctuation tokens.
"""

import re
from typing import List


class WordTokenizer:
    """Regex-based word and punctuation tokenizer."""

    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Convert a text string into a list of word/punctuation tokens.
        
        Args:
            text: Input string.
            
        Returns:
            List of string tokens.
        """
        if not text or not isinstance(text, str):
            return []
        # Case folding
        text_lower = text.strip().lower()
        tokens = cls.TOKEN_PATTERN.findall(text_lower)
        return tokens
