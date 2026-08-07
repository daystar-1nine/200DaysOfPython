# ==============================================================================
# Program    : Unit Test for Addition Function
# Objective  : Verify addition logic across positive, negative, zero, and float values.
# Concept    : unittest.TestCase & assertEqual
# Why Used   : Validates mathematical correctness of addition function.
# ==============================================================================

import unittest

def add(a, b):
    """Returns the sum of numbers a and b."""
    return a + b

# What is used : TestCase class inheriting from unittest.TestCase
class TestAddition(unittest.TestCase):

    # What is used : Test method prefixed with test_
    # Why it is used: unittest framework automatically discovers and executes methods named test_*
    def test_add_positive_integers(self):
        self.assertEqual(add(10, 20), 30)

    def test_add_negative_integers(self):
        self.assertEqual(add(-5, -15), -20)

    def test_add_zero(self):
        self.assertEqual(add(15, 0), 15)

    def test_add_floating_points(self):
        self.assertAlmostEqual(add(2.5, 3.1), 5.6, places=5)

if __name__ == "__main__":
    unittest.main()
