# ==============================================================================
# Program    : Unit Test for Subtraction Function
# Objective  : Verify subtraction logic across positive, negative, and float values.
# Concept    : unittest.TestCase Assertions
# Why Used   : Verifies accurate subtraction calculations.
# ==============================================================================

import unittest

def subtract(a, b):
    """Returns the difference between a and b."""
    return a - b

class TestSubtraction(unittest.TestCase):

    def test_subtract_positive(self):
        self.assertEqual(subtract(25, 10), 15)

    def test_subtract_resulting_negative(self):
        self.assertEqual(subtract(10, 25), -15)

    def test_subtract_negative_numbers(self):
        self.assertEqual(subtract(-10, -5), -5)

    def test_subtract_floating_point(self):
        self.assertAlmostEqual(subtract(5.5, 2.3), 3.2, places=5)

if __name__ == "__main__":
    unittest.main()
