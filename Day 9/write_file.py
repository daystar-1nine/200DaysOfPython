# ==============================================================================
# Program    : Write text to a file
# Objective  : Practice and master write text to a file logic.
# Concept    : Writing text using write() and writelines() methods
# Why Used   : Establishes a file stream for persistent data reading, writing, or appending. Evaluates conditional expressions to control program execution flow.
# ==============================================================================

filename = "written_notes.txt"

content_lines = [
    "Python File Handling Notes:\n",
    "1. Always use 'with' statement.\n",
    "2. Specify utf-8 encoding.\n",
    "3. Close files properly.\n"
]

with open(filename, "w", encoding="utf-8") as file:
    file.write("--- Start of Notes ---\n")
    file.writelines(content_lines)

print(f"Successfully wrote content to '{filename}'.")
