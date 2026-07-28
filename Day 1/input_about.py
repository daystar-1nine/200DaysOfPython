# ==============================================================================
# Program    : Interactive Profile Builder
# Objective  : Practice collecting user inputs and performing explicit type casting.
# Why Used   : input() always returns a string; float() is required to process decimal numbers.
# ==============================================================================

print("-------- MY PROFILE --------")

# Step 1: Collect multiple string inputs from user
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
name = input("Enter your name: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
college = input("Enter your college name: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
course = input("Enter your course: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
company = input("Enter your dream company: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
language = input("Enter your favorite programming language: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
city = input("Enter your city: ")

# Step 2: Collect numeric input and convert to float
# float() converts string representation of decimal number to actual float data type
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
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
