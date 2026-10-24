"""
Day 117 — Coding Challenge 1: Custom Text Chunker with Overlap
Implements sliding-window chunking over raw text preserving character boundaries and overlaps.
"""
from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into chunks of length `chunk_size` with sliding overlap `overlap`.

    Args:
        text: Input string to chunk.
        chunk_size: Maximum character count for each chunk.
        overlap: Overlap character count between consecutive chunks.

    Returns:
        List of string chunks.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap cannot be negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    if not text or not text.strip():
        return []

    clean = text.strip()
    stride = chunk_size - overlap
    chunks = []
    start = 0
    n = len(clean)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(clean[start:end])
        if end == n:
            break
        start += stride

    return chunks


if __name__ == "__main__":
    sample = (
        "Retrieval-Augmented Generation (RAG) is an AI framework for retrieving facts from an external "
        "knowledge base to ground large language models on the most accurate, up-to-date information."
    )
    chunks = chunk_text(sample, chunk_size=50, overlap=10)
    print("=== Challenge 1: Sliding Window Chunking ===")
    print(f"Sample length: {len(sample)} chars")
    print(f"Generated {len(chunks)} chunks with chunk_size=50, overlap=10:")
    for idx, c in enumerate(chunks):
        print(f"  [{idx}]: '{c}' ({len(c)} chars)")
