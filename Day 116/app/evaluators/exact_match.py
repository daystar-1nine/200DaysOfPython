"""
Exact match and token-level accuracy metrics for LLM evaluation.
"""
import re
import string
from typing import Union, List


def normalize_text(text: str) -> str:
    """
    Normalizes text for robust exact match comparison:
    1. Lowercase
    2. Remove punctuation
    3. Remove English articles ('a', 'an', 'the')
    4. Normalize contiguous whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(prediction: str, reference: Union[str, List[str]]) -> float:
    """
    Strict exact match comparison:
    Returns 1.0 if prediction exactly matches reference string (or any of the references).
    Otherwise returns 0.0.
    """
    refs = [reference] if isinstance(reference, str) else reference
    pred_str = prediction.strip()
    for ref in refs:
        if pred_str == ref.strip():
            return 1.0
    return 0.0


def normalized_exact_match(prediction: str, reference: Union[str, List[str]]) -> float:
    """
    Normalized exact match comparison:
    Strips punctuation, articles, and whitespace variations before checking equality.
    """
    refs = [reference] if isinstance(reference, str) else reference
    norm_pred = normalize_text(prediction)
    if not norm_pred:
        return 0.0

    for ref in refs:
        norm_ref = normalize_text(ref)
        if norm_pred == norm_ref:
            return 1.0
    return 0.0


def token_accuracy(prediction: str, reference: Union[str, List[str]]) -> float:
    """
    Computes token-level precision overlap against references:
    Returns the maximum proportion of predicted tokens that appear in the reference tokens.
    """
    refs = [reference] if isinstance(reference, str) else reference
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    best_score = 0.0
    for ref in refs:
        ref_tokens = set(normalize_text(ref).split())
        if not ref_tokens:
            continue
        common = sum(1 for tok in pred_tokens if tok in ref_tokens)
        score = common / len(pred_tokens)
        if score > best_score:
            best_score = score

    return round(best_score, 4)


def char_error_rate(prediction: str, reference: str) -> float:
    """
    Computes Character Error Rate (CER) via Levenshtein edit distance:
      CER = LevenshteinDistance(pred, ref) / max(1, len(ref))
    """
    p = prediction.strip()
    r = reference.strip()
    if not r:
        return 0.0 if not p else 1.0

    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
    for i in range(len(p) + 1):
        dp[i][0] = i
    for j in range(len(r) + 1):
        dp[0][j] = j

    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            if p[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    dist = dp[len(p)][len(r)]
    return round(dist / max(1, len(r)), 4)
