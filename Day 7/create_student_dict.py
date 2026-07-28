# ==============================================================================
# Program    : Create a student dictionary
# Objective  : Practice and master create a student dictionary logic.
# Concept    : Defining key-value pairs and accessing values
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

student = {
    "roll_no": 101,
    "name": "Suraj Sawant",
    "age": 20,
    "course": "Computer Science",
    "cgpa": 8.85
}

print("Student Dictionary:", student)

# Accessing values by key
print(f"Student Name: {student['name']}")
print(f"CGPA: {student['cgpa']}")

# Using .get() for safe access
print(f"Course: {student.get('course')}")
print(f"City (Default fallback): {student.get('city', 'Not Specified')}")
