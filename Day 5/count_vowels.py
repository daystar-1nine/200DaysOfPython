# ==============================================================================
# Program    : Count vowels in a string using a function
# Objective  : Practice and master count vowels in a string using a function logic.
# Concept    : Iteration, membership operators, and character analysis in functions
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

def count_vowels(text):
    """Returns the number of vowels (a, e, i, o, u) in a string"""
    vowel_count = 0
    for char in text:
        if char.lower() in "aeiou":
            vowel_count += 1
    return vowel_count

# Test the function
user_str = input("Enter a word/sentence: ")
print(f"Number of vowels: {count_vowels(user_str)}")
