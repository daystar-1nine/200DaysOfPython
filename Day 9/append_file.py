# Program: Append new text to an existing file
# Concept: Opening file in 'a' mode to append data without overwriting

filename = "append_demo.txt"

# Initialize file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Original Line 1\n")

# Append new text
with open(filename, "a", encoding="utf-8") as f:
    f.write("Appended Line 2\n")
    f.write("Appended Line 3\n")

print(f"--- Contents of '{filename}' after append ---")
with open(filename, "r", encoding="utf-8") as f:
    print(f.read())
