# ==============================================================================
# Program    : Handle Invalid Integer Input
# Objective  : Validate user input type casting to prevent crashing on non-numeric strings.
# Concept    : Exception Handling (try-except ValueError)
# Why Used   : int() raises ValueError if given a non-numeric string like 'abc'.
# ==============================================================================

# What is used : try-except block with ValueError
# Why it is used: Catches invalid literal parsing during int() conversion
# How it works : Traps ValueError and executes fallback error handling logic
try:
    # What is used : int() function
    # Why it is used: Attempts to convert input string into integer
    age = int(input("Enter your age (integer): "))
    print(f"Valid integer accepted: Age = {age}")

except ValueError:
    # What is used : ValueError exception block
    # Why it is used: Informs user that non-numeric characters cannot be converted to integer
    print("Invalid Input Error: Please enter a valid whole number (integer)!")
