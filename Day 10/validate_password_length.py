# ==============================================================================
# Program    : Validate Password Length
# Objective  : Enforce minimum password length rules (at least 8 characters).
# Concept    : Exception Handling & Custom Exception Raising
# Why Used   : Security policies require minimum character length constraints.
# ==============================================================================

# What is used : Custom Exception 'WeakPasswordError'
class WeakPasswordError(Exception):
    pass

def validate_password(pwd):
    # What is used : Built-in len() function
    # Why it is used: Calculates total character count of password string
    if len(pwd) < 8:
        raise WeakPasswordError(f"Security Error: Password too short ({len(pwd)} chars)! Must be at least 8 characters.")
    return "Password security criteria met! Password accepted."

try:
    password = input("Enter new password (min 8 characters): ").strip()
    msg = validate_password(password)
    print(msg)

except WeakPasswordError as err:
    # What is used : Catching custom password exception
    print(err)
