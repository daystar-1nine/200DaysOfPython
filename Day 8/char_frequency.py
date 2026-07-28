# ==============================================================================
# Program    : Count the frequency of each character in a string
# Objective  : Practice and master count the frequency of each character in a string logic.
# Concept    : Dictionary-based frequency counting
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
text = input("Enter a string: ")

frequency = {}


# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for char in text:
    frequency[char] = frequency.get(char, 0) + 1

print("\n--- Character Frequencies ---")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for char, count in frequency.items():
    display_char = repr(char) if char in " \t\n" else char
    print(f"Character {display_char:<6} : {count} time(s)")
