# ==============================================================================
# Program    : Write Data to a JSON File
# Objective  : Save Python dictionary/list records into a formatted JSON file.
# Concept    : JSON File I/O Serialization (json.dump)
# Why Used   : json.dump() writes formatted JSON directly into a file stream.
# ==============================================================================

import json
import os

filename = "sample_created.json"

# What is used : Python list of dictionaries
student_database = [
    {"id": 101, "name": "Suraj", "gpa": 8.85},
    {"id": 102, "name": "Rahul", "gpa": 8.50},
    {"id": 103, "name": "Priya", "gpa": 9.10}
]

# What is used : json.dump(obj, file, indent=4)
# Why it is used: Writes python data structure into JSON file format with clean indentation
with open(filename, "w", encoding="utf-8") as file:
    json.dump(student_database, file, indent=4)

print(f"Successfully wrote {len(student_database)} student records to '{filename}'!")

# Verification: Read back file content
with open(filename, "r", encoding="utf-8") as file:
    print("\nFile Content verification:\n", file.read())

# Cleanup temporary file
if os.path.exists(filename):
    os.remove(filename)
