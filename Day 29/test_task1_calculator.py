# ==============================================================================
# Test Suite : Task 1 Calculator Pytest Suite
# Objective  : Test add, subtract, multiply, and divide functions.
# Concept    : Pytest Native Assertions
# Why Used   : Validates calculator functions against expected results.
# ==============================================================================

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from task1_calculator import add, subtract, multiply, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(10, 4) == 6

def test_multiply():
    assert multiply(3, 4) == 12

def test_divide():
    assert divide(10, 2) == 5.0

class TestTask1Runner(unittest.TestCase):
    def test_all_calculator_methods(self):
        test_add()
        test_subtract()
        test_multiply()
        test_divide()

if __name__ == "__main__":
    unittest.main()
