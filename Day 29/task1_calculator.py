# ==============================================================================
# Program    : Calculator Module for Testing (Task 1)
# Objective  : Basic calculator functions (add, subtract, multiply, divide).
# Concept    : Module to be tested by Pytest
# Why Used   : Provides pure functions for unit testing assertions.
# ==============================================================================

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
