# ==============================================================================
# Program    : Import Math Utilities (Task 1 Main)
# Objective  : Import functions from task1_math_utils module and execute operations.
# Concept    : Module Import (`from module import function`)
# Why Used   : Demonstrates consuming functions defined in external module file.
# ==============================================================================

# What is used : from task1_math_utils import add, subtract
# Why it is used: Imports specific functions directly into local module namespace
from task1_math_utils import add, subtract

def main() -> None:
    print("=== TASK 1: MODULE IMPORT DEMO ===")
    num1, num2 = 25.0, 10.0
    print(f"Addition ({num1} + {num2})    : {add(num1, num2)}")
    print(f"Subtraction ({num1} - {num2}) : {subtract(num1, num2)}")

if __name__ == "__main__":
    main()
