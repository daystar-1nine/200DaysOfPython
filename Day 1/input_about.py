# ==============================================================================
# Program    : Interactive Profile Builder
# Objective  : Practice collecting user inputs and performing explicit type casting.
# Why Used   : input() always returns a string; float() is required to process decimal numbers.
# ==============================================================================

print("-------- MY PROFILE --------")

# Step 1: Collect multiple string inputs from user
name = input("Enter your name: ")
college = input("Enter your college name: ")
course = input("Enter your course: ")
company = input("Enter your dream company: ")
language = input("Enter your favorite programming language: ")
city = input("Enter your city: ")

# Step 2: Collect numeric input and convert to float
# float() converts string representation of decimal number to actual float data type
cgpa = float(input("Enter your CGPA: "))

# Step 3: Display formatted summary
print("\n-------- MY PROFILE --------")
print("Name              :", name)
print("College           :", college)
print("Course            :", course)
print("Dream Company     :", company)
print("Favorite Language :", language)
print("City              :", city)
print("CGPA              :", cgpa)
print("----------------------------")
