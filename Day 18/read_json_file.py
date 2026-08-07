# ==============================================================================
# Program    : Read JSON Records from a File
# Objective  : Read and parse JSON content from a local disk file using json.load().
# Concept    : JSON File I/O Deserialization (json.load)
# Why Used   : json.load() reads JSON data directly from an open file object.
# ==============================================================================

import json
import os

sample_filename = "temp_student.json"

# Helper: Create temporary JSON file for reading demonstration
sample_data = {
    "student_id": 101,
    "name": "Suraj Sawant",
    "course": "Computer Science",
    "grades": {"Python": "A+", "DS": "A"}
}

with open(sample_filename, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=4)

print(f"Created temporary file '{sample_filename}' for reading test.\n")

# What is used : json.load(file_object) inside with context manager
# Why it is used: Reads file stream and parses JSON payload into Python dictionary
with open(sample_filename, "r", encoding="utf-8") as file:
    loaded_data = json.load(file)

print("--- Loaded Data from JSON File ---")
print("Student Name:", loaded_data["name"])
print("Course      :", loaded_data["course"])
print("Python Grade:", loaded_data["grades"]["Python"])

# Cleanup temporary test file
if os.path.exists(sample_filename):
    os.remove(sample_filename)
