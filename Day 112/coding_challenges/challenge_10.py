"""
Day 112 - Coding Challenge 10: Generation Evaluation Metrics
Problem: Implement Distinct-1, Distinct-2, and n-gram repetition rate evaluation functions.
"""
from typing import List, Union, Tuple


def compute_distinct_n(tokens: Union[str, List[str]], n: int = 1) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return round(len(set(ngrams)) / len(ngrams), 4)


def compute_repetition_rate(tokens: Union[str, List[str]], n: int = 3) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    rep_count = len(ngrams) - len(set(ngrams))
    return round(rep_count / len(ngrams), 4)


if __name__ == "__main__":
    words = ["the", "dog", "saw", "the", "cat"]
    d1 = compute_distinct_n(words, n=1)
    d2 = compute_distinct_n(words, n=2)
    rep3 = compute_repetition_rate(words, n=3)

    assert d1 == 4 / 5
    assert d2 == 4 / 4
    assert rep3 == 0.0

    rep_seq = ["ha"] * 10
    assert compute_distinct_n(rep_seq, n=1) == 0.1
    assert compute_repetition_rate(rep_seq, n=1) == 0.9
    print("Challenge 10: PASSED")
