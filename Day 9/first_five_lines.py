# Program: Display only the first 5 lines of a file
# Concept: Line-by-line reading with counter limit or itertools.islice

filename = "multi_lines.txt"

# Prepare sample file with 10 lines
with open(filename, "w", encoding="utf-8") as f:
    for i in range(1, 11):
        f.write(f"This is Line Number {i}\n")

print(f"--- First 5 Lines of '{filename}' ---")
with open(filename, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        if idx > 5:
            break
        print(f"Line {idx}: {line.strip()}")
