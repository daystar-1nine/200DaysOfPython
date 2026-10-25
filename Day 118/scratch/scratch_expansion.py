"""
Challenge 3: Query Expansion, Rewriting, and Multi-Query Decomposition.
"""
import re
from typing import List, Dict, Set


SYNONYM_DICT: Dict[str, List[str]] = {
    "overfitting": ["generalization", "validation loss", "regularization", "dropout"],
    "memory leak": ["garbage collection", "ram consumption", "uncollected objects"],
    "http 404": ["not found", "status code", "endpoint missing", "routing"],
    "inner join": ["sql join", "relational algebra", "matching keys"]
}

PREFIX_PATTERNS = [
    r"^can you (please )?(tell|explain|show) (me )?(what is|how to)?\s*",
    r"^i want to know (about|how)?\s*",
    r"^what is the (meaning of|difference between)\s*"
]


def rewrite_conversational_query(query: str) -> str:
    """Strips conversational noise and filler tokens."""
    cleaned = query.strip()
    for pat in PREFIX_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"[?!.]+$", "", cleaned).strip()
    return cleaned


def expand_technical_query(query: str, max_terms: int = 3) -> str:
    """Appends domain synonyms for detected technical terms."""
    q_lower = query.lower()
    appended: List[str] = []
    seen = set(re.findall(r"\b\w+\b", q_lower))

    for term, syns in SYNONYM_DICT.items():
        if term in q_lower:
            for s in syns[:max_terms]:
                words = s.split()
                if not any(w in seen for w in words):
                    appended.append(s)
                    seen.update(words)

    if not appended:
        return query
    return f"{query} {' '.join(appended)}"


def generate_subqueries(query: str) -> List[str]:
    """Generates multifaceted retrieval subqueries."""
    clean = rewrite_conversational_query(query)
    return [
        clean,
        f"{clean} mechanism principles",
        expand_technical_query(clean, max_terms=2)
    ]


def main():
    print("=== Challenge 3: Query Rewriting & Expansion ===")
    
    test_query = "Can you please tell me how to prevent overfitting in neural networks?"
    print(f"Original User Query:\n  '{test_query}'")

    rewritten = rewrite_conversational_query(test_query)
    print(f"\nRewritten Crisp Query:\n  '{rewritten}'")

    expanded = expand_technical_query(rewritten)
    print(f"\nExpanded Technical Query:\n  '{expanded}'")

    subqueries = generate_subqueries(test_query)
    print(f"\nMulti-Query Decomposition ({len(subqueries)} perspectives):")
    for i, sq in enumerate(subqueries, 1):
        print(f"  {i}. {sq}")

    assert "generalization" in expanded or "regularization" in expanded
    print("\n[SUCCESS] Query rewriting and expansion verified!")


if __name__ == "__main__":
    main()
