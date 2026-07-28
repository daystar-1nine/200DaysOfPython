# ==============================================================================
# Program    : Check if two strings are anagrams
# Objective  : Practice and master check if two strings are anagrams logic.
# Concept    : Sorting characters or character frequency equality
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

def is_anagram(str1, str2):
    # Clean strings: lowercase and remove spaces
    s1 = "".join(c.lower() for c in str1 if c.isalnum())
    s2 = "".join(c.lower() for c in str2 if c.isalnum())
    
    # Compare sorted characters
    return sorted(s1) == sorted(s2)

str1 = input("Enter first string: ")
str2 = input("Enter second string: ")

if is_anagram(str1, str2):
    print(f"[PASS] '{str1}' and '{str2}' ARE Anagrams!")
else:
    print(f"[FAIL] '{str1}' and '{str2}' are NOT Anagrams.")
