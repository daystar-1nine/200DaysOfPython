"""
Train/Validation dataset partitioning, sequence packing, and data leakage detection.
"""
import random
from typing import List, Tuple, Dict, Any, Set
from app.data.dedup import hash_document, get_shingles


def train_val_split(
    documents: List[str],
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Splits documents into train and validation partitions deterministically.
    """
    if not documents:
        return [], []

    rng = random.Random(seed)
    shuffled = list(documents)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_docs = shuffled[:val_count]
    train_docs = shuffled[val_count:]

    return train_docs, val_docs


def detect_leakage(
    train_docs: List[str],
    val_docs: List[str],
    ngram_size: int = 8
) -> Dict[str, Any]:
    """
    Audits training and validation splits for data contamination:
    1. Exact document leakage (identical hash in train and val)
    2. N-gram containment leakage (verbatim sequences appearing in both splits)
    """
    train_hashes: Set[str] = {hash_document(d) for d in train_docs}
    val_hashes: Set[str] = {hash_document(d) for d in val_docs}

    exact_matches = train_hashes.intersection(val_hashes)

    # N-gram shingle overlap
    train_ngrams: Set[str] = set()
    for d in train_docs:
        train_ngrams.update(get_shingles(d, n=ngram_size))

    val_ngrams_total = 0
    val_ngrams_leaked = 0

    for d in val_docs:
        v_shingles = get_shingles(d, n=ngram_size)
        val_ngrams_total += len(v_shingles)
        val_ngrams_leaked += len(v_shingles.intersection(train_ngrams))

    leakage_ratio = (val_ngrams_leaked / val_ngrams_total) if val_ngrams_total > 0 else 0.0

    has_leakage = len(exact_matches) > 0 or leakage_ratio > 0.50

    return {
        "num_train_docs": len(train_docs),
        "num_val_docs": len(val_docs),
        "exact_leaked_documents": len(exact_matches),
        "exact_leak_ratio": round(len(exact_matches) / max(len(val_docs), 1), 4),
        "ngram_overlap_ratio": round(leakage_ratio, 4),
        "has_leakage": has_leakage
    }


def inject_leakage(
    train_docs: List[str],
    val_docs: List[str],
    leakage_fraction: float = 0.50
) -> List[str]:
    """
    Contaminates the training set by intentionally injecting a fraction of validation documents.
    Used for empirical data leakage demonstration experiments.
    """
    num_to_inject = int(len(val_docs) * leakage_fraction)
    leaked_subset = val_docs[:num_to_inject]
    contaminated_train = list(train_docs) + leaked_subset
    return contaminated_train


def pack_sequences(token_ids: List[int], context_length: int) -> Tuple[List[List[int]], int]:
    """
    Packs continuous stream of token IDs into non-overlapping windows of length context_length.
    Discards incomplete trailing tokens.
    Returns (packed_sequences, remainder_tokens_dropped).
    """
    if len(token_ids) < context_length:
        return [], len(token_ids)

    num_full_chunks = len(token_ids) // context_length
    packed = [
        token_ids[i * context_length:(i + 1) * context_length]
        for i in range(num_full_chunks)
    ]
    remainder = len(token_ids) % context_length
    return packed, remainder
