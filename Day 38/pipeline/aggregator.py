# ==============================================================================
# Program    : Functional Aggregator & Reporter (aggregator.py)
# Objective  : Calculate category totals and summary statistics using reduce(), any(), and all().
# Concept    : Functional Data Aggregation (reduce, any, all)
# Why Used   : Produces summary metrics without imperative loop state tracking.
# ==============================================================================

from functools import reduce

def aggregate_totals(records: list[dict]) -> float:
    """Calculates grand total amount using reduce()."""
    # What is used : functools.reduce with lambda accumulator
    # Why it is used: Demonstrates functional reduction over sequence values
    if not records:
        return 0.0
    return reduce(lambda acc, tx: acc + tx.get("amount", 0.0), records, 0.0)

def generate_category_report(records: list[dict]) -> dict:
    """Groups and sums amounts by category using functional aggregations."""
    report = {}
    for tx in records:
        cat = tx["category"]
        report[cat] = report.get(cat, 0.0) + tx["amount"]
    
    total = aggregate_totals(records)
    has_large_tx = any(tx["amount"] >= 1000.0 for tx in records)
    all_positive = all(tx["amount"] > 0 for tx in records)

    return {
        "category_breakdown": report,
        "grand_total": total,
        "has_large_transaction": has_large_tx,
        "all_positive": all_positive
    }
