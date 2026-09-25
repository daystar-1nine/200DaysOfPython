"""
Text encoder module for Day 106: RNNs & Sequential Text Learning.
Translates text strings to integer ID sequences.
"""

from typing import List
from .tokenizer import WordTokenizer
from .vocabulary import Vocabulary


class TextEncoder:
    """Encodes strings into integer token sequences and decodes back to text."""

    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def encode(self, text: str) -> List[int]:
        """Convert a single text string into a list of integer IDs.
        
        Args:
            text: Input string.
            
        Returns:
            List of integer IDs.
        """
        tokens = WordTokenizer.tokenize(text)
        return [self.vocabulary.token_to_id(tok) for tok in tokens]

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode a collection of text strings into lists of integer IDs."""
        return [self.encode(text) for text in texts]

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        """Convert a sequence of token IDs back into readable text string.
        
        Args:
            token_ids: List of integer token IDs.
            skip_special: Whether to omit <PAD> and <UNK> tokens.
            
        Returns:
            Reconstructed text string.
        """
        words = []
        for idx in token_ids:
            word = self.vocabulary.id_to_token(idx)
            if skip_special and word in {Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN}:
                continue
            words.append(word)
        return " ".join(words)
