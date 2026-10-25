"""
Challenge 6: Query-Aware Context Compression.
Prunes filler sentences from retrieved chunks while preserving core facts.
"""
from typing import List, Tuple
import re


def extract_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def score_sentence(sentence: str, query_keywords: List[str]) -> float:
    s_tokens = set(re.findall(r"\b\w+\b", sentence.lower()))
    matches = sum(1 for kw in query_keywords if kw in s_tokens)
    return matches / max(1, len(query_keywords))


def compress_chunk_context(
    query: str,
    raw_chunk: str,
    min_score: float = 0.25
) -> Tuple[str, float]:
    """
    Returns (compressed_text, compression_ratio).
    """
    query_keywords = [w.lower() for w in re.findall(r"\b\w+\b", query)]
    sentences = extract_sentences(raw_chunk)
    kept = []

    for s in sentences:
        score = score_sentence(s, query_keywords)
        if score >= min_score:
            kept.append(s)

    # Fallback to first sentence if all pruned
    if not kept and sentences:
        kept = [sentences[0]]

    compressed = " ".join(kept)
    ratio = (len(raw_chunk) - len(compressed)) / max(1, len(raw_chunk))
    return compressed, max(0.0, ratio)


def main():
    print("=== Challenge 6: Context Compression ===")
    
    query = "How does Python garbage collection handle cyclic references?"
    raw_document = (
        "Python is a beloved programming language created by Guido van Rossum in 1991. "
        "It emphasizes code readability and notable indentation conventions. "
        "The primary memory reclamation mechanism in Python is reference counting. "
        "However, reference counting alone cannot resolve cyclic references between objects. "
        "To handle cyclic references, Python incorporates a generational cyclical garbage collector. "
        "Many developers enjoy using Python for data science and web development. "
        "Frameworks like Django and FastAPI make web servers easy to write."
    )

    compressed, ratio = compress_chunk_context(query, raw_document, min_score=0.2)
    
    print(f"\nOriginal Length: {len(raw_document)} characters")
    print(f"Compressed Length: {len(compressed)} characters")
    print(f"Compression Ratio: {ratio * 100:.1f}% reduction")
    print(f"\nExtracted Evidence Sentences:\n\"{compressed}\"")

    assert "cyclic references" in compressed
    assert "generations" in compressed or "generational" in compressed
    assert "Django" not in compressed
    print("\n[SUCCESS] Context compression successfully pruned filler sentences while preserving core facts!")


if __name__ == "__main__":
    main()
