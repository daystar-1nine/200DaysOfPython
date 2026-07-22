# Program: Basic Login System
# Concept: String Comparison and Logical 'and' Operator

# Step 1: Input username and password from user
username = input("Enter username: ")
password = input("Enter password: ")

# Step 2: Check if credentials match exactly
if username == "admin" and password == "1234":
    print("Login Successful!!")
else:
    print("Invalid Username or Password!!")