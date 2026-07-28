# ==============================================================================
# Program    : Print dictionary values
# Objective  : Practice and master print dictionary values logic.
# Concept    : Using .values() method and loop iteration over values
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.85,
    "city": "Mumbai",
    "country": "India"
}

print("Dictionary Values list:", list(student.values()))

print("\n--- Iterating through Values ---")
for value in student.values():
    print("Value:", value)
