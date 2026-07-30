# ==============================================================================
# Program    : Custom Calculator Helper Module
# Objective  : Provide basic mathematical operations as an importable custom module.
# Concept    : Module Creation, Function Export & Name Guard (__name__ == '__main__')
# Why Used   : Demonstrates creating custom reusable functions in a separate file for external imports.
# ==============================================================================

# What is used : Function definition 'def add(a, b)'
# Why it is used: Provides reusable addition calculation for external callers
# How it works : Takes two numerical operands (a, b) and returns their arithmetic sum
def add(a, b):
    return a + b

# What is used : Function definition 'def subtract(a, b)'
# Why it is used: Provides reusable subtraction calculation for external callers
# How it works : Takes two numerical operands (a, b) and returns their difference (a - b)
def subtract(a, b):
    return a - b

# What is used : Function definition 'def multiply(a, b)'
# Why it is used: Provides reusable multiplication calculation for external callers
# How it works : Takes two numerical operands (a, b) and returns their product (a * b)
def multiply(a, b):
    return a * b

# What is used : Function definition 'def divide(a, b)' with ZeroDivisionError check
# Why it is used: Provides reusable division calculation while preventing division by zero crashes
# How it works : Checks if denominator b == 0; raises ZeroDivisionError if zero, else returns (a / b)
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero in my_calculator module!")
    return a / b

# What is used : Name Guard 'if __name__ == "__main__":'
# Why it is used: Ensures self-testing demo code runs ONLY when file is executed directly, not when imported
# How it works : When run directly, Python sets __name__ = '__main__'. When imported, __name__ = 'my_calculator'
if __name__ == "__main__":
    print("Testing my_calculator module locally...")
    print("10 + 5 =", add(10, 5))
