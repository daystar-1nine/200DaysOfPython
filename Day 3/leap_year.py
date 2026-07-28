# ==============================================================================
# Program    : Leap Year Checker
# Objective  : Determine if a year is a leap year based on Gregorian calendar rules.
# Why Used   : Combines modulo operations: (Year % 4 == 0 and Year % 100 != 0) or (Year % 400 == 0).
# ==============================================================================

# Step 1: Input year
year = int(input("Enter year: "))

# Step 2: Evaluate leap year rules
# A year is leap if divisible by 4 AND not divisible by 100, OR if divisible by 400.
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year} is a Leap Year!")
else:
    print(f"{year} is NOT a Leap Year.")
