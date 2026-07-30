# ==============================================================================
# Program    : Import and Use Custom Module
# Objective  : Demonstrate importing and executing functions from a custom local module.
# Concept    : Custom Module Import (import my_calculator as calc)
# Why Used   : Teaches how to organize code into separate files and import modular functions cleanly.
# ==============================================================================

# What is used : Custom module import with alias 'import my_calculator as calc'
# Why it is used: Imports local file my_calculator.py into current script namespace under short alias 'calc'
# How it works : Python module loader locates my_calculator.py in current directory and executes its scope
import my_calculator as calc

print("=== Custom Module Demonstration ===")

a = 15
b = 5

# What is used : Function calls via module alias 'calc.add(a, b)'
# How it works : Delegates execution to functions defined inside my_calculator.py
add_res = calc.add(a, b)
sub_res = calc.subtract(a, b)
mul_res = calc.multiply(a, b)
div_res = calc.divide(a, b)

print(f"{a} + {b} = {add_res}")
print(f"{a} - {b} = {sub_res}")
print(f"{a} * {b} = {mul_res}")
print(f"{a} / {b} = {div_res}")
