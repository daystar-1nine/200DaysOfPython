# ==============================================================================
# Program    : Mini Project: Password Strength Checker
# Objective  : Practice and master mini project: password strength checker logic.
# Concept    : Checks min length (8), uppercase, lowercase, digit, and special character
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

import string


# What is used : Function definition 'def check_password_strength'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def check_password_strength(password):
    reasons = []

    # Check 1: Minimum 8 characters
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if len(password) < 8:
        reasons.append("Minimum 8 characters required")

    # Check 2: At least one uppercase letter
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not any(char.isupper() for char in password):
        reasons.append("Missing uppercase letter")

    # Check 3: At least one lowercase letter
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not any(char.islower() for char in password):
        reasons.append("Missing lowercase letter")

    # Check 4: At least one digit
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not any(char.isdigit() for char in password):
        reasons.append("Missing digit")

    # Check 5: At least one special character
    special_chars = string.punctuation
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not any(char in special_chars for char in password):
        reasons.append("Missing special character")

    return reasons


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():
    print("====================================")
    print("     Password Strength Checker")
    print("====================================")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    password = input("Enter Password to evaluation: ").strip()

    reasons = check_password_strength(password)

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not reasons:
        print("\nPassword Strength: Strong [PASS]")
    else:
        print("\nPassword Strength: Weak [FAIL]")
        print("Reason(s):")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
        for reason in reasons:
            print(f" - {reason}")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
