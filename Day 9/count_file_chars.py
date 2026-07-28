# Program: Count characters in a file
# Concept: Reading file content and calculating character length

filename = "sample_chars.txt"

# Prepare sample file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Hello World! 123\n")

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Total characters (including spaces & newlines): {len(content)}")
print(f"Total characters (excluding newlines): {len(content.replace('\n', ''))}")
