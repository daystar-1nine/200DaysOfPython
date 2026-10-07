"""
Self-contained BLEU evaluation metric implementation for machine translation and generation.
Implements modified n-gram precision, brevity penalty, and smoothing.
"""
import math
from collections import Counter
from typing import List, Tuple, Dict, Union, Optional

from app.evaluators.exact_match import normalize_text


def compute_ngrams(tokens: List[str], n: int) -> Counter:
    """Extracts counts of all contiguous n-grams from a token sequence."""
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_brevity_penalty(candidate_len: int, reference_len: int) -> float:
    """
    Computes standard BLEU brevity penalty:
      BP = 1.0 if c > r else exp(1 - r / c)
    """
    if candidate_len == 0:
        return 0.0
    if candidate_len > reference_len:
        return 1.0
    return math.exp(1.0 - float(reference_len) / float(candidate_len))


def modified_ngram_precision(
    candidate_tokens: List[str],
    reference_token_lists: List[List[str]],
    n: int
) -> Tuple[int, int]:
    """
    Computes clipped n-gram matches and total candidate n-grams for order n.
    Returns:
      (clipped_matches, total_candidate_ngrams)
    """
    cand_ngrams = compute_ngrams(candidate_tokens, n)
    total_cand = sum(cand_ngrams.values())
    if total_cand == 0:
        return 0, 0

    # Max count across any reference
    max_ref_counts: Dict[Tuple[str, ...], int] = {}
    for ref_tokens in reference_token_lists:
        ref_ngrams = compute_ngrams(ref_tokens, n)
        for ngram, count in ref_ngrams.items():
            max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)

    clipped_matches = 0
    for ngram, count in cand_ngrams.items():
        clipped_matches += min(count, max_ref_counts.get(ngram, 0))

    return clipped_matches, total_cand


def bleu_score(
    prediction: str,
    reference: Union[str, List[str]],
    max_n: int = 4,
    weights: Optional[List[float]] = None,
    smoothing: bool = True
) -> float:
    """
    Calculates sentence-level BLEU score in range [0.0, 1.0].
    Applies Chen & Cherry (2014) smoothing for higher order n-grams with 0 counts.
    """
    refs = [reference] if isinstance(reference, str) else reference
    cand_tokens = normalize_text(prediction).split()
    ref_token_lists = [normalize_text(r).split() for r in refs if r.strip()]

    if not cand_tokens or not ref_token_lists:
        return 0.0

    if weights is None:
        weights = [1.0 / max_n] * max_n
    elif len(weights) != max_n:
        raise ValueError(f"Length of weights ({len(weights)}) must match max_n ({max_n})")

    # Find closest reference length
    cand_len = len(cand_tokens)
    closest_ref_len = min(
        (len(r) for r in ref_token_lists),
        key=lambda r_len: (abs(r_len - cand_len), r_len)
    )

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        clipped, total = modified_ngram_precision(cand_tokens, ref_token_lists, n)
        if total == 0:
            p = 0.0
        elif clipped == 0:
            # Smoothing: if higher order is 0, give small non-zero epsilon if smoothing enabled
            p = (1.0 / (2.0 * total)) if smoothing else 0.0
        else:
            p = float(clipped) / float(total)
        precisions.append(p)

    # Check if any precision is 0.0
    if not smoothing and any(p == 0.0 for p in precisions):
        return 0.0

    # Geometric mean of precisions
    log_sum = 0.0
    for w, p in zip(weights, precisions):
        if p > 0:
            log_sum += w * math.log(p)
        else:
            log_sum += w * -999.0

    bp = compute_brevity_penalty(cand_len, closest_ref_len)
    score = bp * math.exp(log_sum)
    return round(max(0.0, min(1.0, score)), 4)


def sentence_bleu(
    prediction: str,
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Computes individual BLEU-1, BLEU-2, BLEU-3, BLEU-4, and overall cumulative BLEU.
    """
    return {
        "bleu_1": bleu_score(prediction, reference, max_n=1, weights=[1.0]),
        "bleu_2": bleu_score(prediction, reference, max_n=2, weights=[0.5, 0.5]),
        "bleu_3": bleu_score(prediction, reference, max_n=3, weights=[1/3, 1/3, 1/3]),
        "bleu_4": bleu_score(prediction, reference, max_n=4, weights=[0.25, 0.25, 0.25, 0.25]),
        "bleu_overall": bleu_score(prediction, reference, max_n=4)
    }


def corpus_bleu(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> Dict[str, float]:
    """Computes mean corpus BLEU across a dataset."""
    if not predictions:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_4": 0.0, "bleu_overall": 0.0}

    total_1, total_2, total_4, total_overall = 0.0, 0.0, 0.0, 0.0
    n = len(predictions)

    for p, r in zip(predictions, references):
        scores = sentence_bleu(p, r)
        total_1 += scores["bleu_1"]
        total_2 += scores["bleu_2"]
        total_4 += scores["bleu_4"]
        total_overall += scores["bleu_overall"]

    return {
        "bleu_1": round(total_1 / n, 4),
        "bleu_2": round(total_2 / n, 4),
        "bleu_4": round(total_4 / n, 4),
        "bleu_overall": round(total_overall / n, 4)
    }
