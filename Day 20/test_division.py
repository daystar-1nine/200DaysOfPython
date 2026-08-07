# ==============================================================================
# Program    : Unit Test for Division Function & Exception Handling
# Objective  : Verify division math and test ZeroDivisionError exception raising.
# Concept    : Exception Testing via assertRaises
# Why Used   : Verifies that dividing by zero raises ZeroDivisionError exception.
# ==============================================================================

import unittest

def divide(a, b):
    """Returns the quotient of a divided by b."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return a / b

class TestDivision(unittest.TestCase):

    def test_divide_valid(self):
        self.assertEqual(divide(20, 5), 4.0)

    def test_divide_float_result(self):
        self.assertEqual(divide(7, 2), 3.5)

    # What is used : with self.assertRaises(ZeroDivisionError)
    # Why it is used: Asserts that executing divide(10, 0) inside context block raises ZeroDivisionError
    def test_divide_by_zero_exception(self):
        with self.assertRaises(ZeroDivisionError):
            divide(10, 0)

if __name__ == "__main__":
    unittest.main()
