# Program: Write text to a file
# Concept: Writing text using write() and writelines() methods

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
