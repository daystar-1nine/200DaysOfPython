# ==============================================================================
# Program    : Calculate Factorial of a Number
# Objective  : Practice and master calculate factorial of a number logic.
# Concept    : Accumulation (multiplicative), loop structures (for & while), and user input
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# Step 1: Accept positive integer input from user
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter number: "))

# --- Method 1: Using for loop ---
factorial_for = 1

# Loop starts from 1 up to num (inclusive)

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in range(1, num + 1):
    factorial_for = factorial_for * i

print("Factorial using for loop =", factorial_for)


# --- Method 2: Using while loop ---
factorial_while = 1
j = 1

# Loop runs while condition j <= num is True

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while j <= num:
    factorial_while = factorial_while * j
    j += 1

print("Factorial using while loop =", factorial_while)
