# Program: Print dictionary keys
# Concept: Using .keys() method and loop iteration over dictionary keys

student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.85,
    "city": "Mumbai",
    "country": "India"
}

print("Dictionary Keys list:", list(student.keys()))

print("
--- Iterating through Keys ---")
for key in student.keys():
    print("Key:", key)
