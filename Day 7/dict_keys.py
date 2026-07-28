# ==============================================================================
# Program    : Print dictionary keys
# Objective  : Practice and master print dictionary keys logic.
# Concept    : Using .keys() method and loop iteration over dictionary keys
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.85,
    "city": "Mumbai",
    "country": "India"
}

print("Dictionary Keys list:", list(student.keys()))

print("\n--- Iterating through Keys ---")
for key in student.keys():
    print("Key:", key)
