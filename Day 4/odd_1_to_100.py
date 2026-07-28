# ==============================================================================
# Program    : Print Odd Numbers from 1 to 100
# Objective  : Practice and master print odd numbers from 1 to 100 logic.
# Concept    : Loop iteration with condition logic (modulo check for odd numbers)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Odd numbers using for loop ---")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for i in range(1, 100, 2):
    print(i, end=" ")


print("\n--- Odd numbers using while loop ---")
count = 1

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while count <= 100:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if count % 2 != 0:
        print(count, end=" ")
    count += 1
print()
