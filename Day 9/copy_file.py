# ==============================================================================
# Program    : Copy one file into another
# Objective  : Practice and master copy one file into another logic.
# Concept    : Reading source file content and writing to destination file
# Why Used   : Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

source_file = "source.txt"
dest_file = "destination.txt"

# Prepare source file

# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(source_file, "w", encoding="utf-8") as f:
    f.write("This is the original content from source file.\nCopied successfully!\n")


# What is used : Context manager 'with open(...)'
# Why it is used: Guarantees file stream handles are automatically closed after execution
# How it works : Calls __enter__ to open stream and __exit__ to close stream safely
with open(source_file, "r", encoding="utf-8") as src, open(dest_file, "w", encoding="utf-8") as dest:
    dest.write(src.read())

print(f"Copied contents from '{source_file}' to '{dest_file}'.")
