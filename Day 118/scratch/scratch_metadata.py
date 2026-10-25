"""
Challenge 5: Metadata Pre-Filtering and Composable Predicates.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable


@dataclass
class SimpleChunk:
    chunk_id: str
    category: str
    author: str
    token_count: int
    tags: List[str] = field(default_factory=list)


def build_filter(
    category: str = None,
    max_tokens: int = None,
    required_tag: str = None
) -> Callable[[SimpleChunk], bool]:
    """Builds a composable filter predicate."""
    def predicate(c: SimpleChunk) -> bool:
        if category and c.category.lower() != category.lower():
            return False
        if max_tokens and c.token_count > max_tokens:
            return False
        if required_tag and required_tag.lower() not in [t.lower() for t in c.tags]:
            return False
        return True
    return predicate


def main():
    print("=== Challenge 5: Metadata Pre-Filtering ===")
    
    corpus = [
        SimpleChunk("c1", "python", "alice", 120, ["gil", "concurrency"]),
        SimpleChunk("c2", "python", "bob", 450, ["memory", "gc"]),
        SimpleChunk("c3", "dbms", "charlie", 200, ["sql", "join"]),
        SimpleChunk("c4", "python", "alice", 80, ["syntax", "comprehension"]),
        SimpleChunk("c5", "systems", "david", 310, ["networking", "http"])
    ]

    print(f"Total Initial Corpus Chunks: {len(corpus)}")

    # Filter 1: Python only
    py_filter = build_filter(category="python")
    filtered_py = [c for c in corpus if py_filter(c)]
    print(f"\nFiltered (Category='python'): {len(filtered_py)} chunks")
    for c in filtered_py:
        print(f"  [{c.chunk_id}] {c.category} ({c.token_count} tokens) - {c.tags}")

    # Filter 2: Python + max_tokens <= 200 + tag='concurrency'
    strict_filter = build_filter(category="python", max_tokens=200, required_tag="concurrency")
    filtered_strict = [c for c in corpus if strict_filter(c)]
    print(f"\nFiltered (Category='python', max_tokens<=200, tag='concurrency'): {len(filtered_strict)} chunks")
    for c in filtered_strict:
        print(f"  [{c.chunk_id}] {c.category} - {c.tags}")

    assert len(filtered_strict) == 1
    assert filtered_strict[0].chunk_id == "c1"
    print("\n[SUCCESS] Metadata pre-filtering predicates validated!")


if __name__ == "__main__":
    main()
