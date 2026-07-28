# Program: Count words in a file
# Concept: Reading file text and tokenizing words using split()

filename = "sample_words.txt"

# Prepare sample file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Python is an amazing programming language.\nFile handling makes data persistent.\n")

word_count = 0
with open(filename, "r", encoding="utf-8") as f:
    for line in f:
        words = line.split()
        word_count += len(words)

print(f"Total words in '{filename}': {word_count}")
