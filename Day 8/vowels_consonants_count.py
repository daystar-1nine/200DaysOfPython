# ==============================================================================
# Program    : Count vowels and consonants in a string
# Objective  : Practice and master count vowels and consonants in a string logic.
# Concept    : Iteration, membership operators, and character type checking (.isalpha())
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
text = input("Enter a string: ")

vowels_count = 0
consonants_count = 0
vowels = "aeiouAEIOU"


# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for char in text:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if char.isalpha():
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if char in vowels:
            vowels_count += 1
        else:
            consonants_count += 1

print(f"Original Text: {text}")
print(f"Vowels Count    : {vowels_count}")
print(f"Consonants Count: {consonants_count}")
