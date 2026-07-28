# ==============================================================================
# Program    : Sum of Natural Numbers up to N
# Objective  : Practice and master sum of natural numbers up to n logic.
# Concept    : Accumulation (addition), loop bounds, user input processing
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter number: "))

# Using for loop
sum1 = 0

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in range(1, num + 1):
    sum1 = sum1 + i

print("Sum using for =", sum1)

# Using while loop
j = 1
sum2 = 0

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while j <= num:
    sum2 = sum2 + j
    j += 1

print("Sum using while =", sum2)
