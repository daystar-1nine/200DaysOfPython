"""
===============================================================================
DAY 53 — TEST CLEANER MODULE
===============================================================================
This test module verifies string normalization, safe integer/float/date casting,
and record cleaning helper functions.
===============================================================================
"""

from datetime import date
from app.cleaner import safe_int, safe_float, safe_date, clean_record


def test_safe_int_valid_and_invalid():
    """Verify safe_int with valid integers, whitespace, and invalid strings."""
    # What is used: pytest assertions with safe_int.
    # Why it is used: Ensures safe parsing of integer strings without crashing.
    # How it works: Tests "123", "  456  ", "invalid", and None inputs.
    assert safe_int("123") == 123
    assert safe_int("  456  ") == 456
    assert safe_int("invalid") is None
    assert safe_int("invalid", default=0) == 0
    assert safe_int(None) is None


def test_safe_float_valid_and_invalid():
    """Verify safe_float with valid floats, whitespace, and invalid strings."""
    # What is used: pytest assertions with safe_float.
    # Why it is used: Validates safe numeric parsing of price/total floating points.
    # How it works: Tests "55000.50", " 1200 ", "invalid", and None.
    assert safe_float("55000.50") == 55000.50
    assert safe_float(" 1200 ") == 1200.0
    assert safe_float("abc") is None
    assert safe_float(None, default=0.0) == 0.0


def test_safe_date_valid_and_invalid():
    """Verify safe_date parsing ISO YYYY-MM-DD date strings."""
    # What is used: safe_date with strptime ISO format.
    # Why it is used: Validates date parsing for transaction date attributes.
    # How it works: Tests valid ISO string and malformed string formats.
    assert safe_date("2026-09-01") == date(2026, 9, 1)
    assert safe_date("01/09/2026") is None
    assert safe_date("invalid_date") is None
    assert safe_date(None) is None


def test_clean_record_valid():
    """Verify clean_record normalizes raw string inputs into clean dictionary."""
    # What is used: clean_record with dirty raw dictionary row.
    # Why it is used: Validates end-to-end cleaning of a single raw CSV row.
    # How it works: Normalizes strings, parses numbers/date, and asserts field equality.
    raw = {
        "order_id": " 1001 ",
        "customer": " rahul sawant ",
        "product": " laptop ",
        "category": " electronics ",
        "price": " 55000.00 ",
        "quantity": " 1 ",
        "date": " 2026-09-01 ",
    }
    cleaned = clean_record(raw)
    assert cleaned is not None
    assert cleaned["order_id"] == 1001
    assert cleaned["customer"] == "Rahul Sawant"
    assert cleaned["product"] == "Laptop"
    assert cleaned["category"] == "Electronics"
    assert cleaned["price"] == 55000.0
    assert cleaned["quantity"] == 1
    assert cleaned["date"] == date(2026, 9, 1)


def test_clean_record_invalid_row():
    """Verify clean_record returns None for unparseable raw rows."""
    # What is used: clean_record with missing or malformed fields.
    # Why it is used: Ensures invalid raw rows are flagged as None for dropping.
    # How it works: Asserts return is None when price is non-numeric or order_id missing.
    raw_invalid = {"order_id": "abc", "customer": "Rahul", "price": "100"}
    assert clean_record(raw_invalid) is None
