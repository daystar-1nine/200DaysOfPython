# ==============================================================================
# Program    : Print numbers from 100 down to 1 (Reverse Counting)
# Objective  : Practice and master print numbers from 100 down to 1 (reverse counting) logic.
# Concept    : Negative step limits in range(), decrementing counters in while loops
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Reverse counting using for loop ---")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in reversed(range(1,101)):
    print(i, end=" ")
print()

print("\n--- Reverse counting using while loop ---")
j = 100

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while j >= 1:
    print(j, end=" ")
    j -= 1
print()
