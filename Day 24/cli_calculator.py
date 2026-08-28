# ==============================================================================
# Program    : CLI Calculator (Task 1)
# Objective  : Build a CLI calculator supporting positional numbers and --operation flag.
# Concept    : argparse Positional & Optional Arguments with Choices
# Why Used   : Parses positional numbers a and b and computes operation based on --operation flag.
# ==============================================================================

import argparse
import sys

def create_parser() -> argparse.ArgumentParser:
    # What is used : ArgumentParser with description
    # Why it is used: Defines CLI arguments, types, help text, and choices
    parser = argparse.ArgumentParser(description="Command-Line Calculator Tool")
    
    # Positional integer arguments
    parser.add_argument("a", type=float, help="First operand (number)")
    parser.add_argument("b", type=float, help="Second operand (number)")
    
    # Optional --operation argument with restricted choices
    parser.add_argument(
        "--operation",
        type=str,
        default="add",
        choices=["add", "subtract", "multiply", "divide"],
        help="Mathematical operation to perform (default: add)"
    )
    return parser

def calculate(a: float, b: float, operation: str) -> float:
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return a / b
    else:
        raise ValueError(f"Unsupported operation: {operation}")

def main() -> None:
    print("=== TASK 1: CLI CALCULATOR DEMO ===")
    parser = create_parser()
    
    # If no CLI arguments provided during direct test execution, simulate default arguments
    if len(sys.argv) == 1:
        test_args = ["10", "5", "--operation", "add"]
        args = parser.parse_args(test_args)
        print(f"Simulating CLI input: python cli_calculator.py {' '.join(test_args)}")
    else:
        args = parser.parse_args()

    try:
        res = calculate(args.a, args.b, args.operation)
        print(f"Result ({args.operation} {args.a}, {args.b}) : {res}")
    except ZeroDivisionError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
