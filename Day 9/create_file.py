# ==============================================================================
# Program    : Create a New File
# Objective  : Create a text file safely on disk.
# Concept    : Context Manager ('with' statement) & File Writing Modes ('w')
# Why Used   : 'with' automatically manages file resource allocation and guarantees closure.
# ==============================================================================

filename = "sample_created.txt"

# What is used : try-except exception handling block
# Why it is used: Prevents program crash if file permission error occurs
try:
    # What is used : with open(filename, "w", encoding="utf-8") as file
    # Why it is used: Mode 'w' creates file if missing; encoding="utf-8" prevents character corruption
    # How it works : Opens file stream context and assigns stream handle to 'file'
    with open(filename, "w", encoding="utf-8") as file:
        # What is used : file.write() method
        # How it works : Writes string text into the file stream buffer
        file.write("File created successfully!\n")
    print(f"File '{filename}' created successfully.")
except Exception as e:
    print(f"Error creating file: {e}")
