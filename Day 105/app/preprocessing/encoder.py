"""
Text encoder module for Day 105: Neural NLP & Text Classification.
Orchestrates text tokenization and integer encoding through a fitted vocabulary.
"""

from typing import List, Optional
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary


class TextEncoder:
    """Encodes raw text into integer sequence IDs using a Tokenizer and Vocabulary."""

    def __init__(self, tokenizer: Tokenizer, vocabulary: Vocabulary):
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary

    def encode_text(self, text: str) -> List[int]:
        """Convert a single string into a list of token IDs.
        
        Args:
            text: Raw input string.
            
        Returns:
            List of integer IDs.
        """
        tokens = self.tokenizer.tokenize(text)
        return self.vocabulary.encode(tokens)

    def encode_corpus(self, texts: List[str]) -> List[List[int]]:
        """Convert multiple text strings into a list of token ID lists.
        
        Args:
            texts: List of input strings.
            
        Returns:
            List of token ID lists.
        """
        return [self.encode_text(t) for t in texts]

    def decode_ids(self, ids: List[int], skip_special: bool = False) -> str:
        """Decode integer sequence back into space-separated string."""
        tokens = self.vocabulary.decode(ids, skip_special=skip_special)
        return " ".join(tokens)
