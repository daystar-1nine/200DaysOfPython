"""
Deduplication routines for pretraining datasets:
- Exact hash-based deduplication (SHA-256)
- Near-duplicate detection using character/word n-gram Jaccard similarity
"""
import hashlib
from typing import List, Tuple, Set, Dict


def hash_document(doc: str) -> str:
    """Computes SHA-256 hash digest of normalized text."""
    normalized = doc.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exact_deduplicate(documents: List[str]) -> Tuple[List[str], int]:
    """
    Removes exact duplicate documents using SHA-256 hashing.
    Preserves original order of first occurrence.
    Returns (unique_documents, duplicates_removed_count).
    """
    seen_hashes: Set[str] = set()
    unique_docs: List[str] = []
    duplicates_count = 0

    for doc in documents:
        doc_hash = hash_document(doc)
        if doc_hash in seen_hashes:
            duplicates_count += 1
        else:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs, duplicates_count


def get_shingles(text: str, n: int = 3) -> Set[str]:
    """Extracts character n-gram shingles from text."""
    clean = "".join(text.split()).lower()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i + n] for i in range(len(clean) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity: |A & B| / |A | B|."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def near_deduplicate(
    documents: List[str],
    similarity_threshold: float = 0.80,
    shingle_size: int = 3
) -> Tuple[List[str], int]:
    """
    Greedy near-deduplication: filters out documents whose shingle Jaccard similarity
    to any already-accepted document exceeds similarity_threshold.
    Returns (deduplicated_documents, near_duplicates_dropped).
    """
    accepted_docs: List[str] = []
    accepted_shingles: List[Set[str]] = []
    near_dupes_dropped = 0

    for doc in documents:
        doc_shingles = get_shingles(doc, n=shingle_size)
        is_duplicate = False

        for accepted in accepted_shingles:
            sim = jaccard_similarity(doc_shingles, accepted)
            if sim >= similarity_threshold:
                is_duplicate = True
                near_dupes_dropped += 1
                break

        if not is_duplicate:
            accepted_docs.append(doc)
            accepted_shingles.append(doc_shingles)

    return accepted_docs, near_dupes_dropped
