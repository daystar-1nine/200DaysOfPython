# ==============================================================================
# Program    : Transaction Validator (validator.py)
# Objective  : Validate transaction records using all() and filter predicates.
# Concept    : Functional Data Validation (all, filter)
# Why Used   : Ensures raw input datasets adhere to schema rules using functional predicates.
# ==============================================================================

def is_valid_transaction(tx: dict) -> bool:
    """Predicate function asserting required dictionary keys and non-negative amount."""
    required_keys = ["id", "amount", "category"]
    # What is used : all() boolean predicate
    # Why it is used: Verifies every required key is present and amount is positive
    has_keys = all(k in tx for k in required_keys)
    valid_amount = isinstance(tx.get("amount"), (int, float)) and tx.get("amount", -1) >= 0
    return has_keys and valid_amount

def validate_transactions(records: list[dict]) -> list[dict]:
    """Filter records retaining only valid transactions via filter() and lambda."""
    return list(filter(lambda tx: is_valid_transaction(tx), records))
