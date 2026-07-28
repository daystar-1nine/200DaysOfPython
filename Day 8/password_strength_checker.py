# ==============================================================================
# Program    : Mini Project: Password Strength Checker
# Objective  : Practice and master mini project: password strength checker logic.
# Concept    : Checks min length (8), uppercase, lowercase, digit, and special character
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

import string

def check_password_strength(password):
    reasons = []

    # Check 1: Minimum 8 characters
    if len(password) < 8:
        reasons.append("Minimum 8 characters required")

    # Check 2: At least one uppercase letter
    if not any(char.isupper() for char in password):
        reasons.append("Missing uppercase letter")

    # Check 3: At least one lowercase letter
    if not any(char.islower() for char in password):
        reasons.append("Missing lowercase letter")

    # Check 4: At least one digit
    if not any(char.isdigit() for char in password):
        reasons.append("Missing digit")

    # Check 5: At least one special character
    special_chars = string.punctuation
    if not any(char in special_chars for char in password):
        reasons.append("Missing special character")

    return reasons

def main():
    print("====================================")
    print("     Password Strength Checker")
    print("====================================")
    password = input("Enter Password to evaluation: ").strip()

    reasons = check_password_strength(password)

    if not reasons:
        print("\nPassword Strength: Strong [PASS]")
    else:
        print("\nPassword Strength: Weak [FAIL]")
        print("Reason(s):")
        for reason in reasons:
            print(f" - {reason}")

if __name__ == "__main__":
    main()
