# ==============================================================================
# Program    : Search for a word in a file
# Objective  : Practice and master search for a word in a file logic.
# Concept    : Line traversal with substring search and line numbering
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "search_sample.txt"

# Prepare sample file

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "w", encoding="utf-8") as f:
    f.write("Python is easy to learn.\nJava is widely used.\nPython supports object oriented programming.\nC++ is fast.\n")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
target_word = input("Enter word to search in file: ").strip()

found = False
print(f"\n--- Search Results for '{target_word}' ---")

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "r", encoding="utf-8") as f:

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for line_num, line in enumerate(f, start=1):
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if target_word.lower() in line.lower():
            print(f"Line {line_num}: {line.strip()}")
            found = True

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if not found:
    print(f"Word '{target_word}' not found in file.")
