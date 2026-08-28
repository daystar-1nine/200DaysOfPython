# ==============================================================================
# Program    : Greeting CLI Tool (Task 2)
# Objective  : Parse name, optional --age, and boolean --formal flag from command line.
# Concept    : argparse Boolean Flags (store_true) & Optional Parameters
# Why Used   : Customizes output based on flags passed directly via command line.
# ==============================================================================

import argparse
import sys

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Greeting Generator Tool")
    
    # Positional argument
    parser.add_argument("name", type=str, help="Name of person to greet")
    
    # Optional argument
    parser.add_argument("--age", type=int, default=None, help="Age of the person")
    
    # Boolean flag
    # What is used : action="store_true"
    # Why it is used: Sets args.formal to True if --formal is present, False otherwise
    parser.add_argument("--formal", action="store_true", help="Enable formal greeting mode")
    
    return parser

def main() -> None:
    print("=== TASK 2: GREETING CLI DEMO ===")
    parser = create_parser()
    
    if len(sys.argv) == 1:
        test_args = ["Suraj", "--age", "20", "--formal"]
        args = parser.parse_args(test_args)
        print(f"Simulating CLI input: python cli_greet.py {' '.join(test_args)}\n")
    else:
        args = parser.parse_args()

    if args.formal:
        print(f"Good day, {args.name}.")
        if args.age is not None:
            print(f"You are {args.age} years old.")
    else:
        print(f"Hello {args.name}!")
        if args.age is not None:
            print(f"You are {args.age} years old.")

if __name__ == "__main__":
    main()
