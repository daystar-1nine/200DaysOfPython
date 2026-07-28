# ==============================================================================
# Program    : Find the longest word in a sentence
# Objective  : Practice and master find the longest word in a sentence logic.
# Concept    : Tokenization using split() and max() with key=len
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
sentence = input("Enter a sentence: ").strip()

words = sentence.split()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if not words:
    print("No words entered.")
else:
    # Method 1: Using max() function with key=len
    longest = max(words, key=len)
    
    # Method 2: Manual iteration loop
    longest_manual = ""

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for w in words:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if len(w) > len(longest_manual):
            longest_manual = w

    print(f"Longest Word: '{longest}' (Length: {len(longest)})")
