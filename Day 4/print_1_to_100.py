# ==============================================================================
# Program    : Print Numbers from 1 to 100
# Objective  : Practice and master print numbers from 1 to 100 logic.
# Concept    : Simple loop iterations (for and while loops)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Printing 1 to 100 using while loop ---")
count = 1

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while count <= 100:
    print(count, end=" ")
    count += 1
print()

print("\n--- Printing 1 to 100 using for loop ---")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in range(1, 101):
    print(i, end=" ")
print()
