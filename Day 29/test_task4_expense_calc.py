# ==============================================================================
# Test Suite : Task 4 Expense Calculation Pytest Suite
# Objective  : Test calculate_total with empty, single, multiple, and zero amount expenses.
# Concept    : Unit Testing Pure Functions & Edge Cases
# Why Used   : Verifies calculation correctness across edge cases.
# ==============================================================================

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from task4_expense_calc import calculate_total

def test_empty_list():
    assert calculate_total([]) == 0.0

def test_single_expense():
    assert calculate_total([{"amount": 250.0}]) == 250.0

def test_multiple_expenses():
    expenses = [{"amount": 100.0}, {"amount": 200.0}, {"amount": 50.0}]
    assert calculate_total(expenses) == 350.0

def test_zero_amount_expense():
    assert calculate_total([{"amount": 0.0}]) == 0.0

class TestTask4Runner(unittest.TestCase):
    def test_all_scenarios(self):
        test_empty_list()
        test_single_expense()
        test_multiple_expenses()
        test_zero_amount_expense()

if __name__ == "__main__":
    unittest.main()
