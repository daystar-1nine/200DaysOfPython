# ==============================================================================
# Program    : Count words in a sentence
# Objective  : Practice and master count words in a sentence logic.
# Concept    : Using split() method to tokenize sentence by whitespace
# Why Used   : Stores ordered, mutable collections of items allowing dynamic modification. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
sentence = input("Enter a sentence: ").strip()

# Handle empty string edge case
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if not sentence:
    total_words = 0
else:
    words_list = sentence.split()
    total_words = len(words_list)

print(f"Total Words: {total_words}")
