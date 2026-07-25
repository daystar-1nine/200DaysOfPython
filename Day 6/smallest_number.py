# Program: Find the smallest number in a list
# Concept: Tracking minimum value during list iteration vs built-in min()

numbers = [45, 12, 89, 7, 34, 99, 21]
print("Original List:", numbers)

# Method 1: Using built-in min() function
smallest_builtin = min(numbers)
print("Smallest number (using min()):", smallest_builtin)

# Method 2: Manual iteration using a loop
smallest_manual = numbers[0]
for num in numbers:
    if num < smallest_manual:
        smallest_manual = num

print("Smallest number (using loop):", smallest_manual)
