# ==============================================================================
# Program    : Typed Calculator (Task 1)
# Objective  : Implement calculator functions with explicit parameter and return type hints.
# Concept    : Function Type Annotations (int, float)
# Why Used   : Improves code readability, IDE autocompletion, and static type checking.
# ==============================================================================

# What is used : Parameter and return type annotations
# Why it is used: Specifies expected integer parameters and integer return type
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

def multiply(a: int, b: int) -> int:
    return a * b

def divide(a: int, b: int) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return a / b

def main() -> None:
    print("=== TASK 1: TYPED CALCULATOR DEMO ===")
    num1: int = 20
    num2: int = 5

    print(f"Addition ({num1} + {num2})       : {add(num1, num2)}")
    print(f"Subtraction ({num1} - {num2})    : {subtract(num1, num2)}")
    print(f"Multiplication ({num1} * {num2}) : {multiply(num1, num2)}")
    print(f"Division ({num1} / {num2})       : {divide(num1, num2)}")

if __name__ == "__main__":
    main()
