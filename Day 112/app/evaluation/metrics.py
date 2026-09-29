"""
Language modeling evaluation metrics: Perplexity, Distinct-1, Distinct-2.
"""
import math
from typing import List, Dict, Union, Any
import numpy as np


def compute_perplexity(loss: float) -> float:
    """
    Computes perplexity from cross-entropy loss: PPL = exp(loss).
    Clamps loss to prevent floating-point overflow.
    """
    safe_loss = min(float(loss), 50.0)
    return round(math.exp(safe_loss), 4)


def compute_distinct_n(tokens: Union[str, List[str]], n: int = 1) -> float:
    """
    Computes Distinct-N diversity metric:
        Distinct-N = len(unique n-grams) / len(total n-grams)
    Args:
        tokens: String or list of token strings
        n: n-gram order (1 for unigrams, 2 for bigrams)
    Returns:
        Diversity score in range [0.0, 1.0]
    """
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0

    unique_ngrams = set(ngrams)
    return round(len(unique_ngrams) / len(ngrams), 4)


def evaluate_text_diversity(text: str) -> Dict[str, float]:
    """
    Calculates Distinct-1 and Distinct-2 scores for generated text.
    """
    words = text.strip().split()
    chars = list(text)

    # Word-level distinct scores
    d1_word = compute_distinct_n(words, n=1) if len(words) > 0 else 0.0
    d2_word = compute_distinct_n(words, n=2) if len(words) > 1 else 0.0

    # Character-level distinct scores
    d1_char = compute_distinct_n(chars, n=1) if len(chars) > 0 else 0.0
    d2_char = compute_distinct_n(chars, n=2) if len(chars) > 1 else 0.0

    return {
        "distinct_1_word": d1_word,
        "distinct_2_word": d2_word,
        "distinct_1_char": d1_char,
        "distinct_2_char": d2_char
    }
