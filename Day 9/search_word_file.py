# Program: Search for a word in a file
# Concept: Line traversal with substring search and line numbering

filename = "search_sample.txt"

# Prepare sample file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Python is easy to learn.\nJava is widely used.\nPython supports object oriented programming.\nC++ is fast.\n")

target_word = input("Enter word to search in file: ").strip()

found = False
print(f"\n--- Search Results for '{target_word}' ---")
with open(filename, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        if target_word.lower() in line.lower():
            print(f"Line {line_num}: {line.strip()}")
            found = True

if not found:
    print(f"Word '{target_word}' not found in file.")
