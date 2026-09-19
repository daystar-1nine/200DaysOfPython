"""Tokenization utilities for words and character n-grams."""
from typing import List, Tuple
from app.preprocessing.normalization import normalize_text

def word_tokenize(text: str, remove_punct: bool = True) -> List[str]:
    """Tokenizes text into word tokens."""
    cleaned = normalize_text(text, remove_punct=remove_punct)
    return cleaned.split() if cleaned else []

def char_ngrams(text: str, n: int = 3, lowercase: bool = True) -> List[str]:
    """Extracts character n-grams from text."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if lowercase:
        text = text.lower()
    text = text.strip()
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]

def word_ngrams(tokens: List[str], n: int = 2) -> List[Tuple[str, ...]]:
    """Generates word n-grams from a token list."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if not tokens or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
