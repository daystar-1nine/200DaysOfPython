"""
===============================================================================
DAY 53 — DATA CLEANER MODULE
===============================================================================
This module provides string normalization, safe type casting, and raw record
cleaning helper functions.
===============================================================================
"""

from datetime import datetime, date
from typing import Dict, Any, Optional


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert a raw input value into an integer."""
    # What is used: try-except block wrapping int() conversion.
    # Why it is used: Prevents unhandled ValueError exceptions during type casting.
    # How it works: Strips string representation and attempts integer parse; returns default on failure.
    if value is None:
        return default
    try:
        val_str = str(value).strip()
        return int(val_str)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a raw input value into a float."""
    # What is used: try-except block wrapping float() conversion.
    # Why it is used: Prevents application crashes from invalid numeric text strings.
    # How it works: Strips string padding and returns float value or default on parsing error.
    if value is None:
        return default
    try:
        val_str = str(value).strip()
        return float(val_str)
    except (ValueError, TypeError):
        return default


def safe_date(value: Any, date_format: str = "%Y-%m-%d") -> Optional[date]:
    """Safely parse a date string into a datetime.date object."""
    # What is used: datetime.strptime parsing with exception handling.
    # Why it is used: Ensures date strings conform to ISO YYYY-MM-DD standard format.
    # How it works: Attempts strptime conversion and returns date object or None.
    if not value or not isinstance(value, str):
        return None
    try:
        clean_str = value.strip()
        return datetime.strptime(clean_str, date_format).date()
    except (ValueError, TypeError):
        return None


def clean_record(raw_row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Clean and parse a single raw CSV row entry into typed dictionary fields."""
    # What is used: Safe parser helpers and string title-casing.
    # Why it is used: Transforms uncleaned raw string records into standardized data payloads.
    # How it works: Normalizes strings, parses numbers/dates, and returns dict or None if invalid.
    order_id = safe_int(raw_row.get("order_id"))
    customer = str(raw_row.get("customer", "")).strip().title()
    product = str(raw_row.get("product", "")).strip().title()
    category = str(raw_row.get("category", "")).strip().title()
    price = safe_float(raw_row.get("price"))
    quantity = safe_int(raw_row.get("quantity"))
    parsed_date = safe_date(raw_row.get("date"))

    if order_id is None or price is None or quantity is None or parsed_date is None:
        return None

    if not customer or not product or not category:
        return None

    return {
        "order_id": order_id,
        "customer": customer,
        "product": product,
        "category": category,
        "price": price,
        "quantity": quantity,
        "date": parsed_date,
    }
