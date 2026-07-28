# ==============================================================================
# Program    : Count vowels in a string using a function
# Objective  : Practice and master count vowels in a string using a function logic.
# Concept    : Iteration, membership operators, and character analysis in functions
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================


# What is used : Function definition 'def count_vowels'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def count_vowels(text):
    """Returns the number of vowels (a, e, i, o, u) in a string"""
    vowel_count = 0

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for char in text:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if char.lower() in "aeiou":
            vowel_count += 1
    return vowel_count

# Test the function
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
user_str = input("Enter a word/sentence: ")
print(f"Number of vowels: {count_vowels(user_str)}")
