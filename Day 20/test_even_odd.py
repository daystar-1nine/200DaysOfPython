# ==============================================================================
# Program    : Unit Test for Even/Odd Parity Checker
# Objective  : Test boolean parity function using assertTrue and assertFalse assertions.
# Concept    : Boolean Assertions (assertTrue & assertFalse)
# Why Used   : Asserts boolean predicate truths explicitly.
# ==============================================================================

import unittest

def is_even(n):
    """Returns True if n is even, False otherwise."""
    return n % 2 == 0

class TestEvenOdd(unittest.TestCase):

    def test_is_even_true(self):
        # What is used : self.assertTrue(x)
        # Why it is used: Asserts that is_even(4) evaluates to True
        self.assertTrue(is_even(4))
        self.assertTrue(is_even(0))
        self.assertTrue(is_even(-10))

    def test_is_even_false(self):
        # What is used : self.assertFalse(x)
        # Why it is used: Asserts that is_even(7) evaluates to False
        self.assertFalse(is_even(7))
        self.assertFalse(is_even(-5))

if __name__ == "__main__":
    unittest.main()
