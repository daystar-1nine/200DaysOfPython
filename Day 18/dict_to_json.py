# ==============================================================================
# Program    : Convert Python Dictionary to JSON String
# Objective  : Serialize in-memory Python dictionary to formatted JSON string.
# Concept    : JSON Serialization (json.dumps)
# Why Used   : json.dumps() converts Python dictionaries and lists into valid JSON formatted text.
# ==============================================================================

import json

# What is used : Python dictionary containing nested lists and primitive types
# Why it is used: Serves as native in-memory data model
student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.85,
    "skills": ["Python", "C++", "Machine Learning"],
    "is_enrolled": True
}

print("Original Python Dictionary:", student)

# What is used : json.dumps(obj, indent=4)
# Why it is used: Converts dictionary object into JSON formatted string with 4-space indentation
# How it works : Maps Python dict -> JSON object, list -> JSON array, True -> true
json_data = json.dumps(student, indent=4)

print("\n--- Serialized JSON String Output ---")
print(json_data)
