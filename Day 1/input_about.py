# Program: Interactive Profile Builder
# Concept: Taking Multiple Inputs, Type Conversion (float), and Output Formatting

print("-------- MY PROFILE --------")

# Step 1: Collect user input for various fields
name = input("Enter your name: ")
college = input("Enter your college name: ")
course = input("Enter your course: ")
company = input("Enter your dream company: ")
language = input("Enter your favorite programming language: ")
city = input("Enter your city: ")

# Type conversion: float() converts the string input to a decimal number
cgpa = float(input("Enter your CGPA: "))

# Step 2: Display the formatted profile summary
print("\n-------- MY PROFILE --------")
print("Name :", name)
print("College :", college)
print("Course :", course)
print("Dream Company :", company)
print("Favorite Language :", language)
print("City :", city)
print("CGPA :", cgpa)
print("----------------------------")