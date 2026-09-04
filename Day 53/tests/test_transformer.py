"""
===============================================================================
DAY 53 — TEST TRANSFORMER MODULE
===============================================================================
This test module verifies transform_and_deduplicate functions, testing derived
total calculation, invalid record dropping, and duplicate ID deduplication.
===============================================================================
"""

from app.models import Sale
from datetime import date
from app.transformer import transform_and_deduplicate


def test_sale_model_total_property():
    """Verify Sale dataclass computed derived total property."""
    # What is used: Sale dataclass instance creation.
    # Why it is used: Validates derived total property calculation on-the-fly.
    # How it works: 55000.0 * 2 = 110000.0.
    sale = Sale(1001, "Rahul", "Laptop", "Electronics", 55000.0, 2, date(2026, 9, 1))
    assert sale.total == 110000.0


def test_transform_and_deduplicate_success(sample_raw_records):
    """Verify transformation, invalid filtering, and deduplication audit metrics."""
    # What is used: transform_and_deduplicate with sample_raw_records fixture.
    # Why it is used: Validates end-to-end transformation pipeline metrics.
    # How it works: Raw count = 5; 1 duplicate, 2 invalid; valid deduplicated sales = 2.
    sales, invalid_count, duplicate_count = transform_and_deduplicate(sample_raw_records)
    assert len(sales) == 2
    assert invalid_count == 2
    assert duplicate_count == 1
    assert sales[0].order_id == 1001
    assert sales[0].customer == "Rahul"
    assert sales[1].order_id == 1002


def test_transform_and_deduplicate_empty():
    """Verify transform_and_deduplicate with empty raw list."""
    # What is used: transform_and_deduplicate([]).
    # Why it is used: Handles empty list input edge case.
    # How it works: Returns ([], 0, 0).
    sales, invalid, dup = transform_and_deduplicate([])
    assert sales == []
    assert invalid == 0
    assert dup == 0
