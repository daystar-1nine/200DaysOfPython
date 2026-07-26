# Program: Print dictionary values
# Concept: Using .values() method and loop iteration over values

student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.85,
    "city": "Mumbai",
    "country": "India"
}

print("Dictionary Values list:", list(student.values()))

print("
--- Iterating through Values ---")
for value in student.values():
    print("Value:", value)
