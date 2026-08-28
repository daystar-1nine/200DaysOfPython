# ==============================================================================
# Module     : Package Math Utilities (Task 2)
# Objective  : Math functions inside a structured package subfolder.
# Concept    : Package Modules
# Why Used   : Provides mathematical operations as part of the utils package.
# ==============================================================================

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Division by zero error")
    return a / b
