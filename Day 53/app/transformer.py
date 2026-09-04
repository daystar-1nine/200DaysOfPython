"""
===============================================================================
DAY 53 — DATA TRANSFORMER & DEDUPLICATION MODULE
===============================================================================
This module transforms validated dictionary payloads into Sale dataclass instances
and executes deduplication filtering based on unique transaction order_id.
===============================================================================
"""

from typing import List, Dict, Any, Tuple
from app.models import Sale
from app.cleaner import clean_record
from app.validator import validate_sale_dict


def transform_and_deduplicate(
    raw_records: List[Dict[str, str]]
) -> Tuple[List[Sale], int, int]:
    """Process raw records into validated, deduplicated Sale dataclass objects.
    
    Returns:
        Tuple containing (valid_deduplicated_sales, invalid_count, duplicate_count)
    """
    # What is used: Modular processing loop using set() for deduplication.
    # Why it is used: Transforms raw dictionary data into clean model objects while tracking stats.
    # How it works: Cleans, validates, checks order_id against seen set, and creates Sale instances.
    cleaned_valid: List[Dict[str, Any]] = []
    invalid_count = 0

    for raw in raw_records:
        cleaned = clean_record(raw)
        if cleaned is not None and validate_sale_dict(cleaned):
            cleaned_valid.append(cleaned)
        else:
            invalid_count += 1

    seen_ids = set()
    deduplicated_sales: List[Sale] = []
    duplicate_count = 0

    for item in cleaned_valid:
        oid = item["order_id"]
        if oid in seen_ids:
            duplicate_count += 1
        else:
            seen_ids.add(oid)
            sale = Sale(
                order_id=item["order_id"],
                customer=item["customer"],
                product=item["product"],
                category=item["category"],
                price=item["price"],
                quantity=item["quantity"],
                date=item["date"],
            )
            deduplicated_sales.append(sale)

    return (deduplicated_sales, invalid_count, duplicate_count)
