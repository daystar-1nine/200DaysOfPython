# ==============================================================================
# Program    : Count even and odd numbers in a list
# Objective  : Practice and master count even and odd numbers in a list logic.
# Concept    : Loop iteration combined with modulo operator (%)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

numbers = [12, 7, 19, 24, 33, 40, 55, 62, 81, 90]
print("Original List:", numbers)

even_count = 0
odd_count = 0

even_numbers = []
odd_numbers = []


# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for num in numbers:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if num % 2 == 0:
        even_count += 1
        even_numbers.append(num)
    else:
        odd_count += 1
        odd_numbers.append(num)

print(f"Even Numbers ({even_count}): {even_numbers}")
print(f"Odd Numbers ({odd_count}): {odd_numbers}")
