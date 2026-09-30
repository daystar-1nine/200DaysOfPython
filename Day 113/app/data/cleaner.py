"""
Data cleaning, text normalization, and heuristic quality scoring for LLM datasets.
"""
import re
from typing import List, Dict, Any, Tuple


def normalize_text(text: str) -> str:
    """
    Cleans control characters, standardizes unicode whitespace,
    and strips redundant blank lines while preserving paragraph boundaries.
    """
    if not text:
        return ""
    # Strip null bytes and non-printable control characters by replacing with space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Standardize whitespace on each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Collapse multiple consecutive empty lines into a single blank line
    cleaned_text = "\n".join(lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def quality_score(text: str) -> float:
    """
    Computes heuristic quality score in range [0.0, 1.0] for a text document.
    Signals evaluated:
      - Document length (penalizes documents < 20 characters)
      - Alphabetic character ratio (penalizes random numbers, binary dumps, or punctuation soup)
      - Repeated character / symbol patterns (penalizes repetitive spam like 'aaaaaa' or '!!!!!!')
      - HTML / URL density
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    s = text.strip()
    length = len(s)

    # 1. Length score: documents shorter than 20 chars receive severe penalty
    if length < 10:
        return 0.05
    elif length < 25:
        length_score = 0.4
    elif length < 80:
        length_score = 0.8
    else:
        length_score = 1.0

    # 2. Alphabetic ratio: fraction of alphabetic or whitespace characters
    alpha_count = sum(1 for c in s if c.isalpha())
    alpha_ratio = alpha_count / max(length, 1)
    if alpha_ratio >= 0.70:
        alpha_score = 1.0
    elif alpha_ratio >= 0.50:
        alpha_score = 0.7
    else:
        alpha_score = 0.2

    # 3. Repeated character penalty (e.g. 5+ identical consecutive characters)
    has_repetitive_run = bool(re.search(r"(.)\1{4,}", s))
    repetition_score = 0.2 if has_repetitive_run else 1.0

    # 4. URL / HTML tag penalty
    url_count = len(re.findall(r"https?://\S+|www\.\S+", s))
    html_count = len(re.findall(r"<[^>]+>", s))
    markup_penalty = max(0.0, 1.0 - (url_count * 0.4 + html_count * 0.25))

    # Composite weighted score
    total_score = (
        0.30 * length_score +
        0.30 * alpha_score +
        0.20 * repetition_score +
        0.20 * markup_penalty
    )

    # Compound penalty for documents containing both repetitive runs and spam markup
    if has_repetitive_run and (url_count > 0 or html_count > 0):
        total_score *= 0.55

    return round(float(max(0.0, min(1.0, total_score))), 4)


def filter_documents(
    documents: List[str],
    min_quality: float = 0.50
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Filters a collection of documents using quality score.
    Returns (retained_documents, rejection_metadata).
    """
    retained = []
    rejected = []

    for i, doc in enumerate(documents):
        clean_doc = normalize_text(doc)
        score = quality_score(clean_doc)
        if score >= min_quality:
            retained.append(clean_doc)
        else:
            rejected.append({
                "index": i,
                "score": score,
                "length": len(clean_doc),
                "snippet": clean_doc[:40]
            })

    return retained, rejected
