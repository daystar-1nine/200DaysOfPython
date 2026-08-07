# ==============================================================================
# Program    : Convert JSON String to Python Dictionary
# Objective  : Deserialize JSON string payload into a native Python dictionary.
# Concept    : JSON Deserialization (json.loads)
# Why Used   : json.loads() parses JSON formatted text strings into Python dictionaries and lists.
# ==============================================================================

import json

# What is used : JSON formatted string
json_string = '{"name": "Suraj", "age": 20, "course": "Python 200 Days", "active": true}'
print("Raw JSON String:", json_string)

# What is used : json.loads(json_string)
# Why it is used: Deserializes JSON string into native Python dictionary object
# How it works : Parses JSON keys and values, mapping true -> True, null -> None
student_dict = json.loads(json_string)

print("\n--- Deserialized Python Dictionary ---")
print("Python Dict:", student_dict)
print(f"Accessing Key 'name': {student_dict['name']}")
print(f"Accessing Key 'course': {student_dict['course']}")
