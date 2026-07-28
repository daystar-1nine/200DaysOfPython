# ==============================================================================
# Program    : Even or Odd Number Checker
# Objective  : Determine whether a number is even, odd, or zero.
# Why Used   : Uses Modulus operator (% 2) to test divisibility by 2.
# ==============================================================================

# Step 1: Accept integer input
num = int(input("Enter a number: "))

# Step 2: Check if number is 0, even, or odd using if-elif-else ladder
if num == 0:
    print("The number is Zero.")
elif num % 2 == 0:
    print(f"{num} is an Even number.")
else:
    print(f"{num} is an Odd number.")
