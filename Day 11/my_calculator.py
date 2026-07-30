# ==============================================================================
# Program    : Custom Calculator Helper Module
# Objective  : Provide basic mathematical operations as an importable module.
# Concept    : Module Creation & Function Export
# Why Used   : Demonstrates creating custom reusable functions for external imports.
# ==============================================================================

# What is used : Function definition 'def add(a, b)'
# Why it is used: Provides reusable addition calculation
def add(a, b):
    return a + b

# What is used : Function definition 'def subtract(a, b)'
# Why it is used: Provides reusable subtraction calculation
def subtract(a, b):
    return a - b

# What is used : Function definition 'def multiply(a, b)'
# Why it is used: Provides reusable multiplication calculation
def multiply(a, b):
    return a * b

# What is used : Function definition 'def divide(a, b)'
# Why it is used: Provides reusable division calculation with zero check
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero in my_calculator module!")
    return a / b

# What is used : if __name__ == "__main__": block
# Why it is used: Ensures self-testing code runs only when file is executed directly, not when imported
if __name__ == "__main__":
    print("Testing my_calculator module locally...")
    print("10 + 5 =", add(10, 5))
