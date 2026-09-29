"""
Repetition rate and n-gram redundancy analysis for generated text.
"""
from typing import List, Union, Dict, Tuple
from collections import Counter


def repetition_rate(tokens: Union[str, List[str]], n: int = 3) -> float:
    """
    Computes n-gram repetition rate:
        repetition_rate = (total_ngrams - unique_ngrams) / total_ngrams
    Args:
        tokens: String or list of token strings
        n: n-gram length (default: 3)
    Returns:
        Repetition fraction in [0.0, 1.0], where 0 means no repetitions and 1 means complete redundancy.
    """
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    total_ngrams = len(ngrams)
    if total_ngrams == 0:
        return 0.0

    unique_ngrams = len(set(ngrams))
    rep_count = total_ngrams - unique_ngrams
    return round(rep_count / total_ngrams, 4)


def analyze_repetition(text: str, n_values: Tuple[int, ...] = (2, 3, 4)) -> Dict[str, float]:
    """
    Analyzes word-level and character-level repetition across multiple n-gram orders.
    """
    words = text.strip().split()
    chars = list(text)

    results = {}
    for n in n_values:
        results[f"word_rep_rate_{n}gram"] = repetition_rate(words, n=n) if len(words) >= n else 0.0
        results[f"char_rep_rate_{n}gram"] = repetition_rate(chars, n=n) if len(chars) >= n else 0.0

    return results
