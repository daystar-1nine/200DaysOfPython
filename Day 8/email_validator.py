# ==============================================================================
# Program    : Bonus Challenge: Email Validator
# Objective  : Practice and master bonus challenge: email validator logic.
# Concept    : Validates email format (@ count, domain extension, spaces, username)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================


# What is used : Function definition 'def validate_email'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def validate_email(email):
    email = email.strip()

    # Rule 1: No spaces allowed
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if " " in email:
        return False, "Email must not contain spaces."

    # Rule 2: Contains exactly one @
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if email.count("@") != 1:
        return False, "Email must contain exactly one '@' symbol."

    username, domain = email.split("@")

    # Rule 3: Username is not empty
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not username:
        return False, "Username before '@' cannot be empty."

    # Rule 4: Domain must contain a dot and non-empty sub-domain
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if "." not in domain:
        return False, "Domain must contain a valid extension (e.g. .com)."

    # Rule 5: Ends with .com, .in, or .org
    valid_extensions = (".com", ".in", ".org")
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not domain.endswith(valid_extensions):
        return False, f"Domain extension must end with one of {valid_extensions}."

    return True, "Valid Email [PASS]"


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():
    print("====================================")
    print("          EMAIL VALIDATOR           ")
    print("====================================")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    email_input = input("Enter Email: ")

    is_valid, message = validate_email(email_input)

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if is_valid:
        print(f"\n{message}")
    else:
        print("\nInvalid Email [FAIL]")
        print(f"Reason: {message}")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
