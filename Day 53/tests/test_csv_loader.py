"""
===============================================================================
DAY 53 — TEST CSV LOADER MODULE
===============================================================================
This test module verifies load_raw_csv file loading, missing file error handling,
and column header validation.
===============================================================================
"""

import pytest
from pathlib import Path
from app.csv_loader import load_raw_csv


def test_load_raw_csv_success(valid_csv_file: Path):
    """Verify load_raw_csv reads valid CSV file records into dicts."""
    # What is used: load_raw_csv with valid_csv_file fixture.
    # Why it is used: Ensures raw CSV files parse into list of string dictionaries.
    # How it works: Checks length of records list and field values.
    records = load_raw_csv(valid_csv_file)
    assert len(records) == 2
    assert records[0]["customer"] == "Rahul"
    assert records[1]["customer"] == "Aisha"


def test_load_raw_csv_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError raised when loading non-existent CSV file path."""
    # What is used: pytest.raises with FileNotFoundError.
    # Why it is used: Enforces path existence checks for CSV datasets.
    # How it works: Calls load_raw_csv with non-existent path.
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_raw_csv(non_existent)


def test_load_raw_csv_missing_headers(tmp_path: Path):
    """Verify ValueError raised when CSV headers are missing required columns."""
    # What is used: tmp_path to create CSV with incomplete headers.
    # Why it is used: Ensures CSV schema validation.
    # How it works: Writes CSV missing 'price' header; asserts ValueError.
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("order_id,customer,product\n1,Rahul,Laptop\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column headers"):
        load_raw_csv(bad_file)
