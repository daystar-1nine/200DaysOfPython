"""
Day 60 - Pure Python Challenge 3: Age Data Validation
Validates input age, rejecting non-numeric values, negative values, and values above a sensible maximum (120).
"""

# What is used: Import sys module.
# Why it is used: Cross-platform output formatting.
# How it works: Brings sys module into execution scope.
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def validate_age(age: any) -> int:
    """
    Validate age value according to strict domain rules.

    Args:
        age: Age input (int, float, or string).

    Returns:
        int: Cleaned integer age if valid.

    Raises:
        TypeError: If input cannot be converted to integer.
        ValueError: If age is negative (< 0) or above maximum (> 120).
    """
    # What is used: Try-except type conversion.
    # Why it is used: Validates that input is numeric or numeric string.
    # How it works: Attempts int(age) conversion, raising TypeError on failure.
    try:
        age_int = int(age)
    except (ValueError, TypeError):
        raise TypeError(f"Invalid age type: '{age}' is not a numeric value.")

    # What is used: Boundary conditional checks.
    # Why it is used: Rejects logically impossible human ages.
    # How it works: Checks 0 <= age_int <= 120.
    if age_int < 0:
        raise ValueError(f"Invalid age: {age_int} cannot be negative.")
    if age_int > 120:
        raise ValueError(f"Invalid age: {age_int} exceeds sensible maximum of 120.")

    return age_int


def main() -> None:
    test_cases = [25, "30", -5, 150, "twenty", 0, 120]

    print("==================================================")
    print("             AGE VALIDATION AUDIT                 ")
    print("==================================================")
    for test in test_cases:
        try:
            valid_val = validate_age(test)
            print(f"Input: {str(test):<10} -> Validated: {valid_val}")
        except (TypeError, ValueError) as err:
            print(f"Input: {str(test):<10} -> Rejected: {err}")


if __name__ == "__main__":
    main()
