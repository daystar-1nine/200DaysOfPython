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
for item in numbers:
    if item not in unique_list:
        unique_list.append(item)

print("Unique Items (loop, order preserved):", unique_list)
