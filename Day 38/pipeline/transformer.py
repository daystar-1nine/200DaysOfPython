# ==============================================================================
# Program    : Record Transformer (transformer.py)
# Objective  : Normalize dictionary keys and string values using map() and lambda.
# Concept    : Functional Data Transformation (map, lambda)
# Why Used   : Standardizes category names and adds formatted metadata fields.
# ==============================================================================

def transform_records(records: list[dict]) -> list[dict]:
    """Applies clean normalization formatting to transaction records via map()."""
    # What is used : map() with lambda transformation function
    # Why it is used: Applies stateless transformation across all record items
    transform_fn = lambda tx: {
        "id": tx["id"],
        "amount": float(tx["amount"]),
        "category": str(tx["category"]).strip().title(),
        "formatted_amount": f"Rs.{float(tx['amount']):,.2f}"
    }
    return list(map(transform_fn, records))
