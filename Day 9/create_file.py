# ==============================================================================
# Program    : Create a new file
# Objective  : Practice and master create a new file logic.
# Concept    : File creation using 'w' or 'x' mode and context manager
# Why Used   : Establishes a file stream for persistent data reading, writing, or appending.
# ==============================================================================

filename = "sample_created.txt"

try:
    with open(filename, "w", encoding="utf-8") as file:
        file.write("File created successfully!\n")
    print(f"File '{filename}' created successfully.")
except Exception as e:
    print(f"Error creating file: {e}")
