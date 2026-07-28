# ==============================================================================
# Program    : Print Multiplication Table
# Objective  : Practice and master print multiplication table logic.
# Concept    : Basic math operations within loops, loops with start/stop limits
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter Number: "))

print(f"\n--- Multiplication Table for {num} (for loop) ---")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in range(0, 13):
    print(num, "x", i, "=", num * i)

print(f"\n--- Multiplication Table for {num} (while loop) ---")
j = 0

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while j <= 12:
    print(num, "x", j, "=", num * j)
    j += 1
