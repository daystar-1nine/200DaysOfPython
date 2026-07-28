# ==============================================================================
# Program    : Read file contents
# Objective  : Practice and master read file contents logic.
# Concept    : Demonstration of read(), readline(), and readlines() methods
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "read_demo.txt"

# First, prepare a sample file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Line 1: Python Programming\nLine 2: File Handling Basics\nLine 3: Advanced Concepts\n")

print("--- Method 1: read() ---")
with open(filename, "r", encoding="utf-8") as f:
    print(f.read())

print("--- Method 2: readline() ---")
with open(filename, "r", encoding="utf-8") as f:
    print("Line A:", f.readline().strip())
    print("Line B:", f.readline().strip())

print("\n--- Method 3: readlines() ---")
with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()
    print("Lines List:", [line.strip() for line in lines])
