# ==============================================================================
# Program    : User Profile Printer
# Objective  : Demonstrate basic variable declaration, data types, and output formatting.
# Why Used   : Teaches how Python stores user details in variables and displays them 
#              using string concatenation (+) and comma-separated print arguments.
# ==============================================================================

print("-------- MY PROFILE --------")

# Step 1: Input user details
# input() pauses execution to get string input from user
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
name = input("Enter your name: ")
# int() explicitly converts string input to integer
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
age = int(input("Enter your age: "))

# Step 2: Define static profile variables
college = "SJCEM"      # String variable
language = "Python"    # String variable
city = "Mumbai"        # String variable
cgpa = 8.5             # Float variable

# Step 3: Display formatted profile output
# Using '+' joins strings together (requires string operands)
print("\nHello " + name + "!")
# Using ',' handles different data types automatically and inserts spaces
print("You are", age, "years old.")
print("College:", college)
print("Favorite Language:", language)
print("City:", city)
print("CGPA:", cgpa)

print("----------------------------")
