# Program: Update dictionary values and add new entries
# Concept: Mutating dictionary entries and using .update() method

student = {
    "name": "Suraj",
    "age": 20,
    "cgpa": 8.5
}
print("Initial Dictionary:", student)

# Updating existing value directly
student["cgpa"] = 8.85
print("After updating CGPA:", student)

# Adding a new key-value pair
student["city"] = "Mumbai"
print("After adding city:", student)

# Updating multiple entries using .update()
student.update({"age": 21, "github": "daystar-1nine"})
print("After .update():", student)
