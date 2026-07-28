# ==============================================================================
# Program    : Count lines in a file
# Objective  : Practice and master count lines in a file logic.
# Concept    : Iterating line by line to count total lines
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "sample_lines.txt"

# Prepare sample file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Line One\nLine Two\nLine Three\nLine Four\nLine Five\n")

line_count = 0
with open(filename, "r", encoding="utf-8") as f:
    for _ in f:
        line_count += 1

print(f"Total lines in '{filename}': {line_count}")
