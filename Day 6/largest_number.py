# Program: Find the largest number in a list
# Concept: Iterating over list items to track maximum value vs built-in max()

numbers = [23, 89, 12, 56, 99, 45, 78]
print("Original List:", numbers)

# Method 1: Using built-in max() function
largest_builtin = max(numbers)
print("Largest number (using max()):", largest_builtin)

# Method 2: Manual iteration using a loop
largest_manual = numbers[0]
for num in numbers:
    if num > largest_manual:
        largest_manual = num

print("Largest number (using loop):", largest_manual)
