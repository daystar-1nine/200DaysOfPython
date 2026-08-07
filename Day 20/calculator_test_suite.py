# ==============================================================================
# Program    : Calculator Test Suite (Mini Project)
# Objective  : Implement Calculator module functions and test all functions using unittest.
# Concept    : Comprehensive Module Unit Testing
# Why Used   : Tests add, subtract, multiply, divide, power, modulus across all edge cases.
# ==============================================================================

import unittest

# --- CALCULATOR MODULE FUNCTIONS ---
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Division by zero is undefined.")
    return a / b

def power(base, exp):
    return base ** exp

def modulus(a, b):
    if b == 0:
        raise ZeroDivisionError("Modulus by zero is undefined.")
    return a % b


# --- UNIT TEST SUITE ---
class TestCalculatorSuite(unittest.TestCase):

    def test_add_operations(self):
        self.assertEqual(add(15, 25), 40)
        self.assertEqual(add(-10, 5), -5)
        self.assertAlmostEqual(add(1.2, 3.4), 4.6, places=5)

    def test_subtract_operations(self):
        self.assertEqual(subtract(50, 20), 30)
        self.assertEqual(subtract(10, 30), -20)

    def test_multiply_operations(self):
        self.assertEqual(multiply(6, 7), 42)
        self.assertEqual(multiply(-3, 4), -12)
        self.assertEqual(multiply(99, 0), 0)

    def test_divide_operations(self):
        self.assertEqual(divide(100, 4), 25.0)
        self.assertEqual(divide(9, 2), 4.5)
        with self.assertRaises(ZeroDivisionError):
            divide(10, 0)

    def test_power_operations(self):
        self.assertEqual(power(2, 3), 8)
        self.assertEqual(power(5, 0), 1)
        self.assertEqual(power(4, 0.5), 2.0)

    def test_modulus_operations(self):
        self.assertEqual(modulus(10, 3), 1)
        self.assertEqual(modulus(14, 7), 0)
        with self.assertRaises(ZeroDivisionError):
            modulus(10, 0)

if __name__ == "__main__":
    unittest.main()
