# Program: Reverse a string using a function
# Concept: String slicing and string manipulation inside functions

def reverse_string(text):
    """Returns the reversed version of the input string"""
    return text[::-1]

# Test the function
user_str = input("Enter a string: ")
print(f"Reversed string: {reverse_string(user_str)}")
