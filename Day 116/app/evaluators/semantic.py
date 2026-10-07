"""
Semantic similarity and BERTScore-style contextual token alignment metrics.
Uses TF-IDF n-gram vectorization and cosine distance to evaluate semantic meaning
beyond surface-level string matching.
"""
import math
from collections import Counter
from typing import List, Dict, Union, Set
import numpy as np

from app.evaluators.exact_match import normalize_text


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Computes cosine similarity between two sparse feature dictionaries:
      cos(u, v) = (u . v) / (||u|| * ||v||)
    Returns value in [0.0, 1.0].
    """
    if not vec1 or not vec2:
        return 0.0

    dot_product = sum(vec1[k] * vec2[k] for k in vec1 if k in vec2)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return max(0.0, min(1.0, dot_product / (norm1 * norm2)))


def extract_semantic_features(text: str) -> Dict[str, float]:
    """
    Extracts multi-scale semantic features (unigrams, bigrams, and character 3-grams)
    weighted with sub-linear TF scaling.
    """
    norm = normalize_text(text)
    tokens = norm.split()
    features: Counter = Counter()

    # Word unigrams
    for t in tokens:
        features[f"w1_{t}"] += 1.0

    # Word bigrams
    for i in range(len(tokens) - 1):
        features[f"w2_{tokens[i]}_{tokens[i+1]}"] += 1.5

    # Character 3-grams (captures sub-word morphology)
    clean_str = "".join(tokens)
    for i in range(len(clean_str) - 2):
        features[f"c3_{clean_str[i:i+3]}"] += 0.5

    # Sublinear scaling: 1 + log(tf)
    scaled: Dict[str, float] = {}
    for k, count in features.items():
        scaled[k] = 1.0 + math.log(count)

    return scaled


def semantic_similarity(
    prediction: str,
    reference: Union[str, List[str]]
) -> float:
    """
    Computes semantic cosine similarity in range [0.0, 1.0] against one or more references.
    Takes the maximum similarity across reference candidates.
    """
    refs = [reference] if isinstance(reference, str) else reference
    pred_vec = extract_semantic_features(prediction)
    if not pred_vec:
        return 0.0

    best_sim = 0.0
    for r in refs:
        ref_vec = extract_semantic_features(r)
        sim = cosine_similarity(pred_vec, ref_vec)
        if sim > best_sim:
            best_sim = sim

    return round(best_sim, 4)


def bertscore_simulation(
    prediction: str,
    reference: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Simulates the BERTScore token-level greedy matching algorithm:
    Aligns each predicted token with its highest matching reference token (Precision),
    and each reference token with its highest matching candidate token (Recall).
    Returns:
      {"precision": P, "recall": R, "f1": F1}
    """
    refs = [reference] if isinstance(reference, str) else reference
    pred_tokens = normalize_text(prediction).split()

    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_res = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for ref in refs:
        ref_tokens = normalize_text(ref).split()
        if not ref_tokens:
            continue

        # Compute pairwise token similarities using character overlap
        def token_sim(t1: str, t2: str) -> float:
            if t1 == t2:
                return 1.0
            # Jaccard on character bigrams
            bg1 = {t1[i:i+2] for i in range(len(t1) - 1)} if len(t1) > 1 else {t1}
            bg2 = {t2[i:i+2] for i in range(len(t2) - 1)} if len(t2) > 1 else {t2}
            intersection = len(bg1 & bg2)
            union = len(bg1 | bg2)
            return (intersection / union) if union > 0 else 0.0

        # Greedy match for Precision: for each pred token, find max sim in ref
        p_matches = [max(token_sim(p, r) for r in ref_tokens) for p in pred_tokens]
        precision = sum(p_matches) / len(pred_tokens)

        # Greedy match for Recall: for each ref token, find max sim in pred
        r_matches = [max(token_sim(p, r) for p in pred_tokens) for r in ref_tokens]
        recall = sum(r_matches) / len(ref_tokens)

        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 >= best_res["f1"]:
            best_res = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            }

    return best_res
