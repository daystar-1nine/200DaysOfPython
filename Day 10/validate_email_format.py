# ==============================================================================
# Program    : Validate Email Format Using Exceptions
# Objective  : Ensure email addresses contain '@' and '.' characters.
# Concept    : Exception Handling & Custom Exception Raising
# Why Used   : Validates text structural formatting rules before accepting user input.
# ==============================================================================

# What is used : Custom Exception 'InvalidEmailError'
# Why it is used: Signals email validation failure explicitly
class InvalidEmailError(Exception):
    pass

def check_email(email):
    # What is used : Membership operator 'in' and logical AND
    # Why it is used: Checks presence of essential email syntax symbols ('@' and '.')
    if "@" not in email or "." not in email:
        raise InvalidEmailError("Email Error: Address must contain both '@' and '.' domain symbols!")
    if email.startswith("@") or email.endswith("@"):
        raise InvalidEmailError("Email Error: '@' cannot be at start or end of address!")
    return f"Email Address '{email}' is VALID!"

try:
    email_input = input("Enter your email address: ").strip()
    result = check_email(email_input)
    print(result)

except InvalidEmailError as e:
    # What is used : Custom exception handler block
    print(e)
