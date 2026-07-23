# Program: Check if a Year is a Leap Year
# Concept: Logical Operators (and, or) and Modulus calculations for Leap Year rules.
# Rule: A year is a leap year if it is divisible by 400, 
#       OR divisible by 4 but NOT divisible by 100.

# Step 1: Input year from user
year = int(input("Enter Year: "))

# Step 2: Apply the Leap Year logical condition
if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
    print("Leap Year!!")
else:
    print("Not Leap Year!!")