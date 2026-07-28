# ==============================================================================
# Program    : Basic User Login Authentication System
# Objective  : Validate username and password against registered credentials.
# Why Used   : Demonstrates string comparison and logical AND operator in security logic.
# ==============================================================================

# Pre-defined credentials
CORRECT_USER = "admin"
CORRECT_PASS = "admin123"

# Step 1: Accept credentials input
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
username = input("Enter username: ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
password = input("Enter password: ").strip()

# Step 2: Validate both username AND password match
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if username == CORRECT_USER and password == CORRECT_PASS:
    print("\nLogin Successful! Welcome to System Dashboard.")
else:
    print("\nLogin Failed! Invalid username or password.")
