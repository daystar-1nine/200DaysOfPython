"""Tokenization utilities."""
from typing import List
from app.preprocessing.normalization import normalize_text

def tokenize(text: str, remove_punct: bool = True) -> List[str]:
    """
    Tokenizes text into a list of word tokens.
    Handles empty strings, multiple whitespaces, and punctuation.
    """
    cleaned = normalize_text(text, remove_punctuation=remove_punct)
    if not cleaned:
        return []
    return cleaned.split()
