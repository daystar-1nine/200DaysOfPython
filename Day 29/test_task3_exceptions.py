# ==============================================================================
# Test Suite : Task 3 Exception Pytest Suite
# Objective  : Verify divide(10, 0) raises ValueError("Cannot divide by zero").
# Concept    : pytest.raises Exception Assertion
# Why Used   : Ensures code raises expected exceptions on invalid inputs.
# ==============================================================================

import os
import sys
import unittest
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from task1_calculator import divide

def test_divide_by_zero_pytest():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)

class TestTask3Runner(unittest.TestCase):
    def test_divide_by_zero_unittest(self):
        with self.assertRaises(ValueError) as cm:
            divide(10, 0)
        self.assertEqual(str(cm.exception), "Cannot divide by zero")

if __name__ == "__main__":
    unittest.main()
