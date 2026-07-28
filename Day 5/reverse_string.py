# ==============================================================================
# Program    : Reverse a string using a function
# Objective  : Practice and master reverse a string using a function logic.
# Concept    : String slicing and string manipulation inside functions
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

def reverse_string(text):
    """Returns the reversed version of the input string"""
    return text[::-1]

# Test the function
user_str = input("Enter a string: ")
print(f"Reversed string: {reverse_string(user_str)}")
