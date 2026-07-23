# Program: Swap Two Numbers
# Concept: Variable Value Swapping using a Temporary Variable and Pythonic Tuple Unpacking

# Step 1: Input two numbers
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

print(f"\nBefore swapping: First = {num1}, Second = {num2}")

# Method 1: Using a temporary variable (Standard Logic)
temp = num1
num1 = num2
num2 = temp

# Note: Pythonic Way (Alternative without temp):
# num1, num2 = num2, num1

# Step 2: Display swapped values
print("After swapping:")
print("First number =", num1)
print("Second number =", num2)