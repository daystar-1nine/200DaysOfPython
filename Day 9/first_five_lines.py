# ==============================================================================
# Program    : Display only the first 5 lines of a file
# Objective  : Practice and master display only the first 5 lines of a file logic.
# Concept    : Line-by-line reading with counter limit or itertools.islice
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "multi_lines.txt"

# Prepare sample file with 10 lines

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "w", encoding="utf-8") as f:

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for i in range(1, 11):
        f.write(f"This is Line Number {i}\n")

print(f"--- First 5 Lines of '{filename}' ---")

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "r", encoding="utf-8") as f:

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for idx, line in enumerate(f, start=1):
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if idx > 5:
            break
        print(f"Line {idx}: {line.strip()}")
