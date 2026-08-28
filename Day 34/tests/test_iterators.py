# ==============================================================================
# Test Suite : Day 34 Custom Iterators Pytest Suite
# Objective  : Test first item, last item, StopIteration, empty iterators, and bounds errors.
# Concept    : Unit Testing Python Iterator Protocol
# Why Used   : Asserts iteration sequence accuracy and StopIteration raising.
# ==============================================================================

import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from iterators.countdown import CountdownIterator
from iterators.even_numbers import EvenNumberIterator
from iterators.pagination import PaginationIterator
from iterators.transactions import TransactionIterator

# --- CountdownIterator Tests ---
def test_countdown_first_and_last_item():
    it = CountdownIterator(5)
    first = next(it)
    assert first == 5
    items = [first] + list(it)
    assert items == [5, 4, 3, 2, 1]

def test_countdown_stop_iteration():
    it = CountdownIterator(1)
    assert next(it) == 1
    with pytest.raises(StopIteration):
        next(it)

def test_countdown_invalid_input():
    with pytest.raises(ValueError, match="must be at least 1"):
        CountdownIterator(0)

# --- EvenNumberIterator Tests ---
def test_even_numbers_sequence():
    it = EvenNumberIterator(10)
    assert list(it) == [2, 4, 6, 8, 10]

def test_even_numbers_invalid_input():
    with pytest.raises(ValueError, match="must be at least 2"):
        EvenNumberIterator(1)

# --- PaginationIterator Tests ---
def test_pagination_chunks_and_empty():
    items = ["A", "B", "C", "D", "E"]
    pages = list(PaginationIterator(items, page_size=2))
    assert len(pages) == 3
    assert pages[0]["items"] == ["A", "B"]
    assert pages[2]["items"] == ["E"]

def test_pagination_empty_dataset():
    it = PaginationIterator([], page_size=2)
    with pytest.raises(StopIteration):
        next(it)

# --- TransactionIterator Tests ---
def test_transaction_filtering_and_order():
    records = [
        {"id": 1, "amount": 50.0},
        {"id": 2, "amount": 250.0},
        {"id": 3, "amount": 500.0}
    ]
    filtered = list(TransactionIterator(records, min_amount=100.0))
    assert len(filtered) == 2
    assert filtered[0]["id"] == 2
    assert filtered[1]["id"] == 3

def test_transaction_empty_list():
    it = TransactionIterator([])
    with pytest.raises(StopIteration):
        next(it)

class TestIteratorsRunner(unittest.TestCase):
    def test_all_iterators_standalone(self):
        self.assertEqual(list(CountdownIterator(3)), [3, 2, 1])
        self.assertEqual(list(EvenNumberIterator(6)), [2, 4, 6])

if __name__ == "__main__":
    unittest.main()
