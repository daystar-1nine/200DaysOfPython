# ==============================================================================
# Program    : Remove Duplicates Using Set Comprehension
# Objective  : Deduplicate elements and format strings using set comprehension.
# Concept    : Set Comprehensions ({expr for var in iterable})
# Why Used   : Sets automatically enforce element uniqueness while transforming data.
# ==============================================================================

raw_data = ["python", "java", "PYTHON", "C++", "java", "JavaScript", "c++"]
print("Original Raw Data List:", raw_data)

# What is used : Set comprehension '{item.upper() for item in raw_data}'
# Why it is used: Converts elements to uppercase and deduplicates identical entries automatically
# How it works : Normalizes strings to uppercase and inserts into set collection
unique_languages = {item.upper() for item in raw_data}

print("\n--- Deduplicated Unique Set Output ---")
print("Unique Languages Set:", unique_languages)
