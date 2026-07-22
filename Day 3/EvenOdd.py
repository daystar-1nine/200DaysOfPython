# Program: Check if Number is Even, Odd, or Zero
# Concept: If-elif-else ladder, Modulus Operator (%) for even/odd check

# Step 1: Input integer number from the user
num = int(input("Enter number: "))

# Step 2: First check if the number is zero
if num == 0:
    print("Number entered is Zero")
# Step 3: If not zero, check if it's divisible by 2 with no remainder (even number)
elif num % 2 == 0:
    print("Number entered is Even")
# Step 4: Otherwise, the number must be odd
else:
    print("Number entered is Odd")