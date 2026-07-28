# ==============================================================================
# Program    : Remove all spaces from a string
# Objective  : Practice and master remove all spaces from a string logic.
# Concept    : Using replace(" ", "") vs "".join(split())
# Why Used   : Pauses execution to capture interactive user input from standard input.
# ==============================================================================

text = input("Enter a sentence with spaces: ")

# Method 1: Using replace()
no_spaces1 = text.replace(" ", "")

# Method 2: Using split() and join()
no_spaces2 = "".join(text.split())

print("Original Text :", repr(text))
print("Without Spaces:", no_spaces1)
