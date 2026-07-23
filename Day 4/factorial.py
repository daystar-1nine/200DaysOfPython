# Program: Calculate Factorial of a Number
# Concept: Accumulation (multiplicative), loop structures (for & while), and user input

# Step 1: Accept positive integer input from user
num = int(input("Enter number: "))

# --- Method 1: Using for loop ---
factorial_for = 1

# Loop starts from 1 up to num (inclusive)
for i in range(1, num + 1):
    factorial_for = factorial_for * i

print("Factorial using for loop =", factorial_for)


# --- Method 2: Using while loop ---
factorial_while = 1
j = 1

# Loop runs while condition j <= num is True
while j <= num:
    factorial_while = factorial_while * j
    j += 1

print("Factorial using while loop =", factorial_while)
