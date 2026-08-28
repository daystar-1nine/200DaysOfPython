# ==============================================================================
# Test Suite : Day 35 Generators & Large File Processor Pytest Suite
# Objective  : Test reader, filters, data pipeline, and mathematical generators.
# Concept    : Unit Testing Generators & Streaming Pipelines
# Why Used   : Asserts lazy generator behavior, filtering accuracy, and error safety.
# ==============================================================================

import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from processor.reader import read_lines, read_csv_records
from processor.filters import parse_records, filter_positive_amounts, filter_by_category
from processor.pipeline import DataPipeline
from processor.generator_demo import fibonacci_generator, even_numbers_generator, squares_generator

# --- Reader & File Tests ---
def test_read_lines_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        list(read_lines("non_existent_file_999.txt"))

def test_read_lines_valid(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Line 1\nLine 2\n\nLine 3\n")
    lines = list(read_lines(str(f)))
    assert lines == ["Line 1", "Line 2", "Line 3"]

# --- Filter & Pipeline Tests ---
def test_filter_positive_amounts():
    data = [{"id": 1, "amount": 100.0}, {"id": 2, "amount": -50.0}, {"id": 3, "amount": 0.0}]
    filtered = list(filter_positive_amounts(data))
    assert len(filtered) == 1
    assert filtered[0]["id"] == 1

def test_filter_by_category():
    data = [
        {"id": 1, "category": "Food"},
        {"id": 2, "category": "Travel"},
        {"id": 3, "category": "food"}
    ]
    filtered = list(filter_by_category(data, "Food"))
    assert len(filtered) == 2

def test_parse_records_handles_invalid_rows():
    raw_data = [
        {"id": "1", "amount": "250.0", "category": "Food", "description": "Lunch"},
        {"id": "invalid", "amount": "bad_amount"}
    ]
    parsed = list(parse_records(raw_data))
    assert len(parsed) == 1
    assert parsed[0]["amount"] == 250.0

def test_data_pipeline_end_to_end(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("id,amount,category,description\n1,100.0,Food,Lunch\n2,-20.0,Food,Return\n3,500.0,Travel,Ticket\n")
    
    pipeline = DataPipeline(str(f))
    food_records = list(pipeline.process(category="Food"))
    assert len(food_records) == 1
    assert food_records[0]["amount"] == 100.0

# --- Mathematical Generator Tests ---
def test_fibonacci_generator():
    fibs = list(fibonacci_generator(6))
    assert fibs == [0, 1, 1, 2, 3, 5]

def test_squares_generator():
    sqs = list(squares_generator(4))
    assert sqs == [1, 4, 9, 16]

class TestProcessorRunner(unittest.TestCase):
    def test_generators_standalone(self):
        self.assertEqual(list(even_numbers_generator(6)), [2, 4, 6])

if __name__ == "__main__":
    unittest.main()
