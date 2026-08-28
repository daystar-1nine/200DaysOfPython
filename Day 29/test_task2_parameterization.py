# ==============================================================================
# Test Suite : Task 2 Parameterized Pytest Suite
# Objective  : Test add() across multiple dataset parameters.
# Concept    : pytest.mark.parametrize & unittest fallback runner
# Why Used   : Runs multiple test cases cleanly.
# ==============================================================================

import os
import sys
import unittest
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from task1_calculator import add

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (10, 20, 30),
    (-5, 5, 0),
    (0, 0, 0)
])
def test_add_parameterized(a, b, expected):
    assert add(a, b) == expected

class TestTask2Runner(unittest.TestCase):
    def test_parameterized_cases(self):
        cases = [(1, 2, 3), (10, 20, 30), (-5, 5, 0), (0, 0, 0)]
        for a, b, expected in cases:
            self.assertEqual(add(a, b), expected)

if __name__ == "__main__":
    unittest.main()
