# ==============================================================================
# Program    : Count characters in a file
# Objective  : Practice and master count characters in a file logic.
# Concept    : Reading file content and calculating character length
# Why Used   : Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "sample_chars.txt"

# Prepare sample file
with open(filename, "w", encoding="utf-8") as f:
    f.write("Hello World! 123\n")

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

print(f"Total characters (including spaces & newlines): {len(content)}")
print(f"Total characters (excluding newlines): {len(content.replace('\n', ''))}")
