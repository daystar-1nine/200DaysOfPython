# ==============================================================================
# Program    : Swap Two Numbers
# Objective  : Exchange the values stored in two variables.
# Why Used   : Demonstrates Python's tuple unpacking syntax for 1-line variable swapping 
#              without needing a temporary third variable.
# ==============================================================================

# Step 1: Input two values
a = input("Enter first value (a): ")
b = input("Enter second value (b): ")

print(f"\n--- Before Swapping ---")
print(f"a = {a}, b = {b}")

# Step 2: Swap values using Python's tuple unpacking: a, b = b, a
# Python packs (b, a) into a tuple in memory, then unpacks into a and b
a, b = b, a

# Step 3: Display swapped values
print(f"\n--- After Swapping ---")
print(f"a = {a}, b = {b}")
