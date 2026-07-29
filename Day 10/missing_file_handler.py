# ==============================================================================
# Program    : Handle Missing File Exception
# Objective  : Safely open files without crashing if target file does not exist.
# Concept    : Exception Handling (try-except FileNotFoundError)
# Why Used   : Attempting to open a non-existent file in read mode ('r') raises FileNotFoundError.
# ==============================================================================

filename = input("Enter file name to open: ").strip()

# What is used : try-except FileNotFoundError handler
# Why it is used: Traps I/O file opening errors if file path is missing on disk
# How it works : Monitors open() call; jumps to except block if OS reports file missing
try:
    # What is used : with open() context manager
    # Why it is used: Ensures file is automatically closed after reading
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
        print("\n--- File Content ---")
        print(content)

except FileNotFoundError:
    # What is used : FileNotFoundError exception handler
    # Why it is used: Informs user gracefully that requested file was not found on path
    print(f"File Error: The file '{filename}' does not exist on this path!")
