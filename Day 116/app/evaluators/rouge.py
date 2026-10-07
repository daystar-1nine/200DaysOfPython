"""
Self-contained ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) evaluation metric implementation.
Computes overlap precision, recall, and harmonic F1 scores.
"""
from typing import List, Dict, Union
from collections import Counter

from app.evaluators.exact_match import normalize_text


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts counts of all contiguous n-grams from a token sequence."""
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """
    Computes Longest Common Subsequence (LCS) length between two token lists
    using dynamic programming in O(M*N) time and O(min(M, N)) space.
    """
    if not seq1 or not seq2:
        return 0

    m, n = len(seq1), len(seq2)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def rouge_n(
    prediction: str,
    reference: Union[str, List[str]],
    n: int = 1
) -> Dict[str, float]:
    """
    Computes ROUGE-N precision, recall, and F1 score against one or more references.
    Takes the maximum F1 match across reference candidates.
    """
    refs = [reference] if isinstance(reference, str) else reference
    cand_tokens = normalize_text(prediction).split()

    if not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    cand_ngrams = compute_ngrams(cand_tokens, n)
    total_cand = sum(cand_ngrams.values())
    if total_cand == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_res = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for ref in refs:
        ref_tokens = normalize_text(ref).split()
        if not ref_tokens:
            continue
        ref_ngrams = compute_ngrams(ref_tokens, n)
        total_ref = sum(ref_ngrams.values())
        if total_ref == 0:
            continue

        overlap = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in cand_ngrams.items())

        p = overlap / total_cand
        r = overlap / total_ref
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        if f1 >= best_res["f1"]:
            best_res = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4)
            }

    return best_res


def rouge_l(
    prediction: str,
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Computes ROUGE-L precision, recall, and F1 score using Longest Common Subsequence.
    """
    refs = [reference] if isinstance(reference, str) else reference
    cand_tokens = normalize_text(prediction).split()

    if not cand_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_res = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for ref in refs:
        ref_tokens = normalize_text(ref).split()
        if not ref_tokens:
            continue

        lcs = compute_lcs_length(cand_tokens, ref_tokens)
        p = lcs / len(cand_tokens)
        r = lcs / len(ref_tokens)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        if f1 >= best_res["f1"]:
            best_res = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4)
            }

    return best_res


def compute_rouge_all(
    prediction: str,
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, and ROUGE-L summary F1 scores."""
    r1 = rouge_n(prediction, reference, n=1)
    r2 = rouge_n(prediction, reference, n=2)
    rl = rouge_l(prediction, reference)

    return {
        "rouge_1": r1["f1"],
        "rouge_2": r2["f1"],
        "rouge_l": rl["f1"],
        "rouge_1_recall": r1["recall"],
        "rouge_1_precision": r1["precision"]
    }


def corpus_rouge(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> Dict[str, float]:
    """Computes mean ROUGE-1, ROUGE-2, and ROUGE-L across a list of examples."""
    if not predictions:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    total_r1, total_r2, total_rl = 0.0, 0.0, 0.0
    n = len(predictions)

    for p, r in zip(predictions, references):
        scores = compute_rouge_all(p, r)
        total_r1 += scores["rouge_1"]
        total_r2 += scores["rouge_2"]
        total_rl += scores["rouge_l"]

    return {
        "rouge_1": round(total_r1 / n, 4),
        "rouge_2": round(total_r2 / n, 4),
        "rouge_l": round(total_rl / n, 4)
    }
