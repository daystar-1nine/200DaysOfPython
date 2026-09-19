"""Vocabulary construction and token frequency module."""
from collections import Counter
from typing import List, Dict, Set

def build_vocabulary(tokenized_docs: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """
    Builds sorted vocabulary dictionary mapping word -> index.
    Filters out words occurring fewer than min_freq times.
    """
    counter = Counter()
    for doc in tokenized_docs:
        counter.update(doc)
        
    filtered_words = sorted([word for word, count in counter.items() if count >= min_freq])
    return {word: idx for idx, word in enumerate(filtered_words)}

def get_word_frequencies(tokenized_docs: List[List[str]]) -> Counter:
    """Returns frequency Counter across all tokenized documents."""
    counter = Counter()
    for doc in tokenized_docs:
        counter.update(doc)
    return counter
