# ==============================================================================
# Program    : Append new text to an existing file
# Objective  : Practice and master append new text to an existing file logic.
# Concept    : Opening file in 'a' mode to append data without overwriting
# Why Used   : Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "append_demo.txt"

# Initialize file

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "w", encoding="utf-8") as f:
    f.write("Original Line 1\n")

# Append new text

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "a", encoding="utf-8") as f:
    f.write("Appended Line 2\n")
    f.write("Appended Line 3\n")

print(f"--- Contents of '{filename}' after append ---")

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(filename, "r", encoding="utf-8") as f:
    print(f.read())
