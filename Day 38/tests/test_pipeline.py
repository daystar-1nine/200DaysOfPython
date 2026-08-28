# ==============================================================================
# Test Suite : Day 38 Functional Data Transformation Pipeline Pytest Suite
# Objective  : Test validation, filtering, transformation, sorting, and reduction aggregations.
# Concept    : Unit Testing Functional Pipelines
# Why Used   : Asserts functional purity, filter criteria, and calculation accuracy.
# ==============================================================================

import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pipeline.validator import validate_transactions
from pipeline.filters import filter_high_value
from pipeline.transformer import transform_records
from pipeline.sorter import sort_by_key
from pipeline.aggregator import aggregate_totals, generate_category_report
from main_pipeline import run_pipeline

# 1. Test Empty Data
def test_pipeline_empty_data():
    report = run_pipeline([], min_amount=500.0)
    assert report["grand_total"] == 0.0
    assert report["records"] == []
    assert report["has_large_transaction"] is False

# 2. Test Single Record Below Threshold
def test_single_record_below_threshold():
    raw = [{"id": 1, "amount": 200.0, "category": "Food"}]
    report = run_pipeline(raw, min_amount=500.0)
    assert len(report["records"]) == 0

# 3. Test Single Record Above Threshold
def test_single_record_above_threshold():
    raw = [{"id": 1, "amount": 750.0, "category": "Food"}]
    report = run_pipeline(raw, min_amount=500.0)
    assert len(report["records"]) == 1
    assert report["grand_total"] == 750.0

# 4. Test Validation Filters Invalid & Negative Records
def test_validation_filters_invalid():
    raw = [
        {"id": 1, "amount": 600.0, "category": "Food"},
        {"id": 2, "amount": -100.0, "category": "Bad"},
        {"missing_amount": True}
    ]
    valid = validate_transactions(raw)
    assert len(valid) == 1
    assert valid[0]["id"] == 1

# 5. Test Filtering High Value Threshold
def test_filter_high_value():
    records = [{"amount": 100.0}, {"amount": 500.0}, {"amount": 1200.0}]
    filtered = filter_high_value(records, min_amount=500.0)
    assert len(filtered) == 2

# 6. Test Data Transformation Normalization
def test_transform_records():
    raw = [{"id": 1, "amount": 250, "category": " food "}]
    transformed = transform_records(raw)
    assert transformed[0]["category"] == "Food"
    assert transformed[0]["formatted_amount"] == "Rs.250.00"

# 7. Test Sorter Descending Order
def test_sorter_descending():
    records = [{"amount": 200}, {"amount": 1000}, {"amount": 500}]
    sorted_recs = sort_by_key(records, key="amount", reverse=True)
    assert [r["amount"] for r in sorted_recs] == [1000, 500, 200]

# 8. Test Reduce Aggregation Total
def test_aggregate_totals():
    records = [{"amount": 100.0}, {"amount": 250.0}, {"amount": 350.0}]
    total = aggregate_totals(records)
    assert total == 700.0

# 9. Test Category Report Aggregation
def test_category_report():
    records = [
        {"amount": 500.0, "category": "Food"},
        {"amount": 700.0, "category": "Food"},
        {"amount": 1200.0, "category": "Travel"}
    ]
    report = generate_category_report(records)
    assert report["category_breakdown"]["Food"] == 1200.0
    assert report["category_breakdown"]["Travel"] == 1200.0
    assert report["grand_total"] == 2400.0
    assert report["has_large_transaction"] is True
    assert report["all_positive"] is True

class TestPipelineRunner(unittest.TestCase):
    def test_pipeline_standalone(self):
        records = [{"id": 1, "amount": 600.0, "category": "Food"}]
        self.assertEqual(len(validate_transactions(records)), 1)

if __name__ == "__main__":
    unittest.main()
