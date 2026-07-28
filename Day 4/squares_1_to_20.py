# ==============================================================================
# Program    : Print Squares of numbers from 1 to 20
# Objective  : Practice and master print squares of numbers from 1 to 20 logic.
# Concept    : Number power operations, range limits, increment operations
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Squares from 1 to 20 using for loop ---")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in range(1, 21):
    print(i, "square =", i * i)

print("\n--- Squares from 1 to 20 using while loop ---")
j = 1

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while j <= 20:
    print(j, "square =", j * j)
    j += 1
