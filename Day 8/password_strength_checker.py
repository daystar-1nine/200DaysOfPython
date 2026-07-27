# Mini Project: Password Strength Checker
# Features: Checks min length (8), uppercase, lowercase, digit, and special character

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
        print("
Password Strength: Strong ✅")
    else:
        print("
Password Strength: Weak ❌")
        print("Reason(s):")
        for reason in reasons:
            print(f" - {reason}")

if __name__ == "__main__":
    main()
