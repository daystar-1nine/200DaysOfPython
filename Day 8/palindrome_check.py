# Program: Check if a string is a palindrome
# Concept: String normalization (case-insensitive & whitespace-stripping) and comparison

def is_palindrome(text):
    # Clean string: lowercase and remove non-alphanumeric characters
    cleaned = "".join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]

input_str = input("Enter a word or sentence: ")

if is_palindrome(input_str):
    print(f"✅ '{input_str}' is a Palindrome!")
else:
    print(f"❌ '{input_str}' is NOT a Palindrome.")
