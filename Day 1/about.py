# Program: User Profile Printer
# Concept: Basic Variables, Data Types, and String Concatenation vs Comma Formatting

print("-------- MY PROFILE --------")

# Step 1: Input user details (string and integer casting)
name = input("Enter your name: ")
age = int(input("Enter your age: "))

# Step 2: Define static variables
college = "SJCEM"
language = "Python"
city = "Mumbai"
cgpa = 8.5

# Step 3: Print formatted profile details
print("\nHello " + name + "!")
print("You are", age, "years old.")
print("College:", college)
print("Favorite Language:", language)
print("City:", city)
print("CGPA:", cgpa)

print("----------------------------")