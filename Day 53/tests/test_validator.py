"""
===============================================================================
DAY 53 — TEST VALIDATOR MODULE
===============================================================================
This test module verifies business validation constraints for sales records.
===============================================================================
"""

from datetime import date
from app.validator import validate_sale_dict


def test_validate_sale_dict_valid():
    """Verify validate_sale_dict returns True for compliant record dictionaries."""
    # What is used: validate_sale_dict with valid record dictionary.
    # Why it is used: Ensures valid records pass boundary checks cleanly.
    # How it works: Evaluates compliant fields; asserts True.
    valid_dict = {
        "order_id": 1001,
        "customer": "Rahul",
        "product": "Laptop",
        "category": "Electronics",
        "price": 55000.0,
        "quantity": 1,
        "date": date(2026, 9, 1),
    }
    assert validate_sale_dict(valid_dict) is True


def test_validate_sale_dict_invalid_order_id():
    """Verify validate_sale_dict rejects negative or zero order IDs."""
    # What is used: validate_sale_dict with invalid order_id values (0 and -5).
    # Why it is used: Enforces positive integer constraint on primary order ID.
    # How it works: Expects False when order_id <= 0.
    bad_dict = {
        "order_id": 0,
        "customer": "Rahul",
        "product": "Laptop",
        "category": "Electronics",
        "price": 55000.0,
        "quantity": 1,
        "date": date(2026, 9, 1),
    }
    assert validate_sale_dict(bad_dict) is False


def test_validate_sale_dict_invalid_price():
    """Verify validate_sale_dict rejects negative prices."""
    # What is used: validate_sale_dict with negative price (-100.0).
    # Why it is used: Prevents negative financial transaction prices.
    # How it works: Expects False when price < 0.0.
    bad_dict = {
        "order_id": 1001,
        "customer": "Rahul",
        "product": "Laptop",
        "category": "Electronics",
        "price": -100.0,
        "quantity": 1,
        "date": date(2026, 9, 1),
    }
    assert validate_sale_dict(bad_dict) is False


def test_validate_sale_dict_invalid_quantity():
    """Verify validate_sale_dict rejects zero or negative quantities."""
    # What is used: validate_sale_dict with quantity 0.
    # Why it is used: Ensures transaction quantity is at least 1 unit.
    # How it works: Expects False when quantity <= 0.
    bad_dict = {
        "order_id": 1001,
        "customer": "Rahul",
        "product": "Laptop",
        "category": "Electronics",
        "price": 55000.0,
        "quantity": 0,
        "date": date(2026, 9, 1),
    }
    assert validate_sale_dict(bad_dict) is False


def test_validate_sale_dict_empty_fields():
    """Verify validate_sale_dict rejects empty string customer or product."""
    # What is used: validate_sale_dict with empty customer name "".
    # Why it is used: Prevents anonymous or incomplete customer transaction records.
    # How it works: Expects False when customer name is empty.
    bad_dict = {
        "order_id": 1001,
        "customer": "",
        "product": "Laptop",
        "category": "Electronics",
        "price": 55000.0,
        "quantity": 1,
        "date": date(2026, 9, 1),
    }
    assert validate_sale_dict(bad_dict) is False
