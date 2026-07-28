# Program: Create a new file
# Concept: File creation using 'w' or 'x' mode and context manager

filename = "sample_created.txt"

try:
    with open(filename, "w", encoding="utf-8") as file:
        file.write("File created successfully!\n")
    print(f"File '{filename}' created successfully.")
except Exception as e:
    print(f"Error creating file: {e}")
