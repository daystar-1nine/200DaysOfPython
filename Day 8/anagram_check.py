# ==============================================================================
# Program    : Check if two strings are anagrams
# Objective  : Practice and master check if two strings are anagrams logic.
# Concept    : Sorting characters or character frequency equality
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================


# What is used : Function definition 'def is_anagram'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def is_anagram(str1, str2):
    # Clean strings: lowercase and remove spaces
    s1 = "".join(c.lower() for c in str1 if c.isalnum())
    s2 = "".join(c.lower() for c in str2 if c.isalnum())
    
    # Compare sorted characters
    return sorted(s1) == sorted(s2)

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
str1 = input("Enter first string: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
str2 = input("Enter second string: ")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if is_anagram(str1, str2):
    print(f"[PASS] '{str1}' and '{str2}' ARE Anagrams!")
else:
    print(f"[FAIL] '{str1}' and '{str2}' are NOT Anagrams.")
