# ==============================================================================
# Program    : Functional Record Sorter (sorter.py)
# Objective  : Sort records by dynamic key using sorted() and key=lambda.
# Concept    : Functional Higher-Order Sorting (sorted, key=lambda)
# Why Used   : Orders collections flexibly by amount, category, or ID.
# ==============================================================================

def sort_by_key(records: list[dict], key: str = "amount", reverse: bool = True) -> list[dict]:
    """Sorts dictionary records by key attribute using sorted()."""
    # What is used : sorted() with key=lambda extractor
    # Why it is used: Higher-order sorting preserving original list immutability
    return sorted(records, key=lambda tx: tx.get(key, 0), reverse=reverse)
