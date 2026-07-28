# ==============================================================================
# Program    : Count lines in a file
# Objective  : Practice and master count lines in a file logic.
# Concept    : Iterating line by line to count total lines
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "sample_lines.txt"

# Prepare sample file

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "w", encoding="utf-8") as f:
    f.write("Line One\nLine Two\nLine Three\nLine Four\nLine Five\n")

line_count = 0

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "r", encoding="utf-8") as f:

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for _ in f:
        line_count += 1

print(f"Total lines in '{filename}': {line_count}")
