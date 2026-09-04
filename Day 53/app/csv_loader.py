"""
===============================================================================
DAY 53 — CSV LOADER MODULE
===============================================================================
This module provides safe CSV file loading capabilities for raw datasets.
===============================================================================
"""

import csv
from pathlib import Path
from typing import List, Dict


def load_raw_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file stream and return a list of raw string dictionaries."""
    # What is used: Path.exists check and csv.DictReader context manager.
    # Why it is used: Ingests raw CSV file records while maintaining header mappings.
    # How it works: Checks file existence, validates content non-emptiness, and reads rows.
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Raw CSV dataset file not found at path: '{path}'")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    records: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_fields = {"order_id", "customer", "product", "category", "price", "quantity", "date"}
        if not reader.fieldnames or not required_fields.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV dataset missing required column headers: {required_fields}")

        for row in reader:
            records.append(dict(row))

    return records
