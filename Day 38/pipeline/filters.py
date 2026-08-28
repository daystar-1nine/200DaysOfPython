# ==============================================================================
# Program    : Transaction Filter (filters.py)
# Objective  : Filter records by minimum threshold using filter() and lambda.
# Concept    : Functional Filtering (filter, lambda)
# Why Used   : Extracts high-value items exceeding min_amount threshold.
# ==============================================================================

def filter_high_value(records: list[dict], min_amount: float = 500.0) -> list[dict]:
    """Retains transactions where amount >= min_amount."""
    # What is used : filter() with lambda predicate
    # Why it is used: Higher-order functional filtering without manual loop syntax
    return list(filter(lambda tx: tx.get("amount", 0.0) >= min_amount, records))
