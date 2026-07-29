# ==============================================================================
# Program    : Handle Multiple Exceptions Together
# Objective  : Demonstrate multiple specialized except handlers for robust error recovery.
# Concept    : Multiple Except Blocks (ValueError, ZeroDivisionError, Exception)
# Why Used   : Different runtime errors require different corrective feedback messages.
# ==============================================================================

# What is used : Multiple except clauses attached to single try block
# Why it is used: Catches specific exception types sequentially from top to bottom
try:
    a_str = input("Enter numerator (integer): ")
    b_str = input("Enter denominator (integer): ")

    # Potential ValueError if string parsing fails
    a = int(a_str)
    b = int(b_str)

    # Potential ZeroDivisionError if b is 0
    quotient = a / b
    print(f"Calculation Result: {a} / {b} = {quotient:.2f}")

except ValueError:
    # What is used : Handling string-to-int parsing failure
    print("Input Error: One or both inputs were not valid integers!")

except ZeroDivisionError:
    # What is used : Handling division by zero failure
    print("Math Error: Cannot divide an integer by zero!")

except Exception as generic_err:
    # What is used : Wildcard base Exception fallback handler
    # Why it is used: Catches any unexpected exception not explicitly caught above
    print(f"Unexpected Error Occurred: {generic_err}")
