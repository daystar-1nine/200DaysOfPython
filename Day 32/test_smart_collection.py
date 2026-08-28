# ==============================================================================
# Test Suite : Smart Collection Pytest & Unittest Suite
# Objective  : Test 10+ edge cases for dunder protocols in SmartCollection.
# Concept    : Unit Testing Dunder Protocol Implementations
# Why Used   : Asserts correctness of container emulation and operator overloading.
# ==============================================================================

import os
import sys
import unittest
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from smart_collection import SmartCollection

# 1. Test empty collection
def test_empty_collection():
    col = SmartCollection()
    assert len(col) == 0
    assert str(col) == "SmartCollection([])"

# 2. Test adding item
def test_adding_item():
    col = SmartCollection()
    col.add("Python")
    assert len(col) == 1
    assert col[0] == "Python"

# 3. Test removing item
def test_removing_item():
    col = SmartCollection(["A", "B", "C"])
    col.remove("B")
    assert len(col) == 2
    assert "B" not in col

# 4. Test length
def test_length():
    col = SmartCollection([1, 2, 3, 4, 5])
    assert len(col) == 5

# 5. Test indexing and slicing
def test_indexing_and_slicing():
    col = SmartCollection(["Zero", "One", "Two", "Three"])
    assert col[0] == "Zero"
    assert col[-1] == "Three"
    sliced = col[1:3]
    assert len(sliced) == 2
    assert sliced[0] == "One"

# 6. Test contains (__contains__)
def test_contains():
    col = SmartCollection(["Docker", "Kubernetes"])
    assert "Docker" in col
    assert "Jenkins" not in col

# 7. Test iteration (__iter__)
def test_iteration():
    items = ["Apple", "Banana", "Cherry"]
    col = SmartCollection(items)
    iterated = [item for item in col]
    assert iterated == items

# 8. Test equality (__eq__)
def test_equality():
    col1 = SmartCollection([1, 2, 3])
    col2 = SmartCollection([1, 2, 3])
    col3 = SmartCollection([1, 2, 4])
    assert col1 == col2
    assert col1 != col3

# 9. Test invalid index raises IndexError
def test_invalid_index():
    col = SmartCollection(["Item"])
    with pytest.raises(IndexError):
        _ = col[99]

# 10. Test duplicate items
def test_duplicate_items():
    col = SmartCollection(["A", "A", "B"])
    assert len(col) == 3
    assert col[0] == "A"
    assert col[1] == "A"

# 11. Test collection addition (__add__)
def test_collection_addition():
    col1 = SmartCollection(["A", "B"])
    col2 = SmartCollection(["C", "D"])
    combined = col1 + col2
    assert len(combined) == 4
    assert list(combined) == ["A", "B", "C", "D"]

class TestSmartCollectionRunner(unittest.TestCase):
    def test_all_smart_collection_features(self):
        col = SmartCollection(["Python", "SQL"])
        self.assertEqual(len(col), 2)
        self.assertEqual(col[0], "Python")
        self.assertTrue("SQL" in col)
        col.add("Docker")
        self.assertEqual(len(col), 3)
        col.remove("SQL")
        self.assertFalse("SQL" in col)

if __name__ == "__main__":
    unittest.main()
