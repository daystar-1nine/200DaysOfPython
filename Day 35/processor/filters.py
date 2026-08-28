# ==============================================================================
# Program    : Generator Filters & Parsers (filters.py)
# Objective  : Stream parsing and filtering generators for records and values.
# Concept    : Generator Pipeline Filtering Stages
# Why Used   : Transforms or filters items in streaming pipelines lazily.
# ==============================================================================

def parse_records(records):
    """Generator converting CSV dictionary values to typed objects."""
    for r in records:
        try:
            yield {
                "id": int(r.get("id", 0)),
                "amount": float(r.get("amount", 0.0)),
                "category": r.get("category", "").strip(),
                "description": r.get("description", "").strip()
            }
        except (ValueError, TypeError):
            # Safely skip malformed rows
            continue

def filter_positive_amounts(records):
    """Generator filtering records where amount > 0."""
    for r in records:
        if r.get("amount", 0.0) > 0:
            yield r

def filter_by_category(records, target_category: str):
    """Generator filtering records matching a specific category."""
    cat_clean = target_category.strip().lower()
    for r in records:
        if r.get("category", "").strip().lower() == cat_clean:
            yield r
