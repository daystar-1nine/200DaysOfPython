# Program: Copy one file into another
# Concept: Reading source file content and writing to destination file

source_file = "source.txt"
dest_file = "destination.txt"

# Prepare source file
with open(source_file, "w", encoding="utf-8") as f:
    f.write("This is the original content from source file.\nCopied successfully!\n")

with open(source_file, "r", encoding="utf-8") as src, open(dest_file, "w", encoding="utf-8") as dest:
    dest.write(src.read())

print(f"Copied contents from '{source_file}' to '{dest_file}'.")
