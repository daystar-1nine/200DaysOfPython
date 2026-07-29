# ==============================================================================
# Program    : Validate Age Input Using Custom Exceptions
# Objective  : Validate human age range limits (1 to 120 years).
# Concept    : Custom Exception Class (Inheriting from Exception)
# Why Used   : Custom exceptions provide domain-specific error handling for age validation.
# ==============================================================================

# What is used : Custom Exception Class 'InvalidAgeError(Exception)'
# Why it is used: Inherits from base Exception to create specialized domain exception type
# How it works : Overrides __init__ to store custom error message
class InvalidAgeError(Exception):
    """Raised when the age is out of acceptable bounds (1 to 120)."""
    pass

def validate_age(age):
    if age <= 0 or age > 120:
        # What is used : raise statement with custom exception class
        raise InvalidAgeError(f"Age Error: '{age}' is outside valid range (1 to 120)!")
    return f"Age '{age}' successfully validated!"

try:
    user_age = int(input("Enter your age (1-120): "))
    msg = validate_age(user_age)
    print(msg)

except InvalidAgeError as err:
    # What is used : Catching custom exception type 'InvalidAgeError'
    print(err)

except ValueError:
    # What is used : Catching non-integer input parsing errors
    print("Type Error: Age must be a valid integer number!")
