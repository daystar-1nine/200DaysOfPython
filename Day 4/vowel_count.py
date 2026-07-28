# ==============================================================================
# Program    : Count Vowels in a Word
# Objective  : Practice and master count vowels in a word logic.
# Concept    : String traversal, character membership check, case sensitivity handling, loop counters
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
word = input("Enter a word: ")

# Using for loop (traversing characters directly)
count = 0

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for ch in word:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if ch.lower() in "aeiou":
        count += 1

print("Number of vowels (using for) =", count)

# Using while loop (traversing using indices)
count = 0
j = 0

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
while j < len(word):
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if word[j].lower() in "aeiou":
        count += 1
    j += 1

print("Number of vowels (using while) =", count)
