# ==============================================================================
# Program    : Import from Package (Task 2 Main)
# Objective  : Import functions from utils package modules.
# Concept    : Package Imports (`from package.module import function`)
# Why Used   : Demonstrates organizing code into structured packages.
# ==============================================================================

import os
import sys

# Ensure task2_package_demo directory is in Python path for testing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# What is used : from utils.math_utils import multiply, divide
# Why it is used: Accesses submodules from the utils package
from utils.math_utils import multiply, divide
from utils.string_utils import reverse_string, capitalize_words

def main() -> None:
    print("=== TASK 2: PACKAGE IMPORT DEMO ===")
    print(f"Multiply (5 * 4)   : {multiply(5, 4)}")
    print(f"Divide (20 / 4)    : {divide(20, 4)}")
    print(f"Reverse String     : {reverse_string('Python')}")
    print(f"Capitalize Words   : {capitalize_words('python project architecture')}")

if __name__ == "__main__":
    main()
