"""Tokenization utilities for Word2Vec."""
import re
import string
from typing import List

def tokenize_text(text: str) -> List[str]:
    """
    Cleans and tokenizes text into lowercase word tokens.
    Handles punctuation and whitespace normalization.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
        
    text = text.lower().strip()
    # Strip punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    tokens = re.sub(r'\s+', ' ', text).strip().split()
    return tokens

def tokenize_corpus(corpus: List[str]) -> List[List[str]]:
    """Tokenizes a collection of documents."""
    return [tokenize_text(doc) for doc in corpus if doc]
