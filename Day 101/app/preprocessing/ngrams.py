"""N-gram generation module."""
from typing import List, Tuple

def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generates sequence of n-grams from token list.
    Example: ['I', 'love', 'python'], n=2 -> [('I', 'love'), ('love', 'python')]
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not tokens or len(tokens) < n:
        return []
        
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
