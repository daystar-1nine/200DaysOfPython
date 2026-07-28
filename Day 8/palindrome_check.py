# ==============================================================================
# Program    : Check if a string is a palindrome
# Objective  : Practice and master check if a string is a palindrome logic.
# Concept    : String normalization (case-insensitive & whitespace-stripping) and comparison
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

def is_palindrome(text):
    # Clean string: lowercase and remove non-alphanumeric characters
    cleaned = "".join(char.lower() for char in text if char.isalnum())
    return cleaned == cleaned[::-1]

input_str = input("Enter a word or sentence: ")

if is_palindrome(input_str):
    print(f"[PASS] '{input_str}' is a Palindrome!")
else:
    print(f"[FAIL] '{input_str}' is NOT a Palindrome.")
