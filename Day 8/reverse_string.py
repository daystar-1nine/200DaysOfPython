# Program: Reverse a string
# Concept: Slicing [::-1], reversed() function, and loop concatenation

user_str = input("Enter a string to reverse: ")

# Method 1: String Slicing (Recommended)
reversed_slice = user_str[::-1]
print("Reversed (Slicing):", reversed_slice)

# Method 2: Using reversed() and join()
reversed_join = "".join(reversed(user_str))
print("Reversed (reversed() + join()):", reversed_join)

# Method 3: Using a loop
reversed_loop = ""
for char in user_str:
    reversed_loop = char + reversed_loop
print("Reversed (Loop):", reversed_loop)
