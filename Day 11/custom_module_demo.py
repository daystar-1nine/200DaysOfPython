# ==============================================================================
# Program    : Import and Use Custom Module
# Objective  : Demonstrate importing and executing functions from a custom local module.
# Concept    : Custom Module Import (my_calculator.py)
# Why Used   : Teaches how to split code into separate files and import functions cleanly.
# ==============================================================================

# What is used : Custom module import 'import my_calculator as calc'
# Why it is used: Imports local file my_calculator.py with alias 'calc'
# How it works : Python searches current directory, finds my_calculator.py, and loads its functions
import my_calculator as calc

print("=== Custom Module Demonstration ===")

a = 15
b = 5

# What is used : calc.add(a, b)
# How it works : Calls add() function defined in my_calculator.py
add_res = calc.add(a, b)
sub_res = calc.subtract(a, b)
mul_res = calc.multiply(a, b)
div_res = calc.divide(a, b)

print(f"{a} + {b} = {add_res}")
print(f"{a} - {b} = {sub_res}")
print(f"{a} * {b} = {mul_res}")
print(f"{a} / {b} = {div_res}")
