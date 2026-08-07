# ==============================================================================
# Program    : Unit Test for Factorial Function
# Objective  : Verify factorial calculations for 0, 1, positive numbers, and ValueError for negative inputs.
# Concept    : Mathematical Boundary & Exception Testing
# Why Used   : Checks factorial base cases (0! = 1, 1! = 1) and negative input error handling.
# ==============================================================================

import unittest

def factorial(n):
    """Computes factorial of non-negative integer n."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

class TestFactorial(unittest.TestCase):

    def test_factorial_base_cases(self):
        self.assertEqual(factorial(0), 1)
        self.assertEqual(factorial(1), 1)

    def test_factorial_positive(self):
        self.assertEqual(factorial(5), 120)
        self.assertEqual(factorial(6), 720)

    def test_factorial_negative_exception(self):
        with self.assertRaises(ValueError):
            factorial(-5)

if __name__ == "__main__":
    unittest.main()
