# ==============================================================================
# Program    : Remove duplicates from a list
# Objective  : Practice and master remove duplicates from a list logic.
# Concept    : Using set() vs maintaining original order with a loop
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

numbers = [10, 20, 10, 30, 40, 20, 50, 30, 60]
print("Original List with Duplicates:", numbers)

# Method 1: Using set() (Order is not preserved)
unique_set = list(set(numbers))
print("Unique Items (set, order lost):", unique_set)

# Method 2: Loop iteration (Preserves original order)
unique_list = []

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for item in numbers:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if item not in unique_list:
        unique_list.append(item)

print("Unique Items (loop, order preserved):", unique_list)
