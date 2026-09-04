"""
===============================================================================
DAY 53 — PYTEST FIXTURES MODULE
===============================================================================
This module provides Pytest fixtures for Sale dataclass instances, raw dictionary
records, valid CSV files, and temporary path fixtures.
===============================================================================
"""

import pytest
from pathlib import Path
from datetime import date
from typing import List, Dict
from app.models import Sale


@pytest.fixture
def sample_sales() -> List[Sale]:
    """Fixture providing a list of test Sale dataclass instances."""
    return [
        Sale(1001, "Rahul", "Laptop", "Electronics", 55000.0, 1, date(2026, 9, 1)),
        Sale(1002, "Aisha", "Mouse", "Electronics", 1200.0, 2, date(2026, 9, 1)),
        Sale(1003, "Rohan", "Keyboard", "Electronics", 2500.0, 1, date(2026, 9, 2)),
        Sale(1004, "Sneha", "Chair", "Furniture", 7000.0, 2, date(2026, 9, 2)),
        Sale(1005, "Karan", "Desk", "Furniture", 15000.0, 1, date(2026, 9, 3)),
    ]


@pytest.fixture
def sample_raw_records() -> List[Dict[str, str]]:
    """Fixture providing a list of raw dictionary rows including messy entries."""
    return [
        {"order_id": "1001", "customer": " rahul ", "product": "laptop", "category": "electronics", "price": "55000", "quantity": "1", "date": "2026-09-01"},
        {"order_id": "1002", "customer": "Aisha", "product": "Mouse", "category": "Electronics", "price": "1200", "quantity": "2", "date": "2026-09-01"},
        {"order_id": "1002", "customer": "Aisha", "product": "Mouse", "category": "Electronics", "price": "1200", "quantity": "2", "date": "2026-09-01"},  # duplicate
        {"order_id": "1003", "customer": "Rohan", "product": "Chair", "category": "Furniture", "price": "-500", "quantity": "1", "date": "2026-09-02"},   # invalid price
        {"order_id": "1004", "customer": "Sneha", "product": "Desk", "category": "Furniture", "price": "15000", "quantity": "0", "date": "2026-09-03"},    # invalid quantity
    ]


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a temporary valid raw CSV file."""
    csv_file = tmp_path / "valid_sales.csv"
    content = (
        "order_id,customer,product,category,price,quantity,date\n"
        "1001,Rahul,Laptop,Electronics,55000,1,2026-09-01\n"
        "1002,Aisha,Mouse,Electronics,1200,2,2026-09-01\n"
    )
    csv_file.write_text(content, encoding="utf-8")
    return csv_file


@pytest.fixture
def invalid_csv_file(tmp_path: Path) -> Path:
    """Fixture creating a temporary CSV file with invalid row data."""
    csv_file = tmp_path / "invalid_sales.csv"
    content = (
        "order_id,customer,product,category,price,quantity,date\n"
        "1001,Rahul,Laptop,Electronics,-5000,1,2026-09-01\n"
    )
    csv_file.write_text(content, encoding="utf-8")
    return csv_file
