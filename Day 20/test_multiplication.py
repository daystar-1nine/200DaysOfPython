# ==============================================================================
# Program    : Unit Test for Multiplication Function
# Objective  : Verify multiplication logic across positive, negative, and zero values.
# Concept    : Multiplication Edge Cases
# Why Used   : Ensures multiplication by zero, negative, and positive factors returns correct results.
# ==============================================================================

import unittest

def multiply(a, b):
    """Returns the product of a and b."""
    return a * b

class TestMultiplication(unittest.TestCase):

    def test_multiply_positive(self):
        self.assertEqual(multiply(4, 5), 20)

    def test_multiply_by_zero(self):
        self.assertEqual(multiply(100, 0), 0)

    def test_multiply_negative(self):
        self.assertEqual(multiply(-4, 5), -20)
        self.assertEqual(multiply(-4, -5), 20)

if __name__ == "__main__":
    unittest.main()
