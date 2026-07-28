# ==============================================================================
# Program    : Reverse a String
# Objective  : Generate reversed sequence of characters from input string.
# Concept    : String Slicing [::-1], reversed() & Loop Concatenation
# Why Used   : Demonstrates multiple algorithms for string reversal in Python.
# ==============================================================================

user_str = input("Enter a string to reverse: ")

# What is used : Extended String Slicing [::-1]
# Why it is used: [start:stop:step] with step -1 steps backwards from end to start
# How it works : Creates new string reading characters in reverse order
reversed_slice = user_str[::-1]
print("Reversed (Slicing):", reversed_slice)

# What is used : join() method combined with built-in reversed() iterator
# Why it is used: reversed() yields characters in reverse; "".join() concatenates them
reversed_join = "".join(reversed(user_str))
print("Reversed (reversed() + join()):", reversed_join)

# What is used : Manual loop with prepending concatenation
# How it works : Prepends each character to front of accumulator string
reversed_loop = ""
for char in user_str:
    reversed_loop = char + reversed_loop
print("Reversed (Loop):", reversed_loop)
