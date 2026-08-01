# ==============================================================================
# Program    : Email and Phone Validator (Mini Project)
# Objective  : Interactive validation tool for user email and phone inputs using regex.
# Concept    : Anchor & Quantifier Validation Patterns
# Why Used   : Validates email and Indian phone formats with ASCII indicators ([VALID] / [INVALID]).
# ==============================================================================

import re

# What is used : Compiled regex patterns for performance
# Email: standard username@domain.extension format
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
# Phone: 10-digit Indian mobile number starting with 6-9
PHONE_PATTERN = re.compile(r"^[6-9]\d{9}$")

def validate_email(email):
    return bool(EMAIL_PATTERN.match(email.strip()))

def validate_phone(phone):
    return bool(PHONE_PATTERN.match(phone.strip()))

def main():
    print("==========================================================")
    print("            EMAIL & PHONE NUMBER VALIDATOR                ")
    print("==========================================================")
    
    email_input = input("Enter Email Address: ").strip()
    phone_input = input("Enter Phone Number : ").strip()

    email_status = "[VALID]" if validate_email(email_input) else "[INVALID]"
    phone_status = "[VALID]" if validate_phone(phone_input) else "[INVALID]"

    print("\n------------------ VALIDATION RESULTS ------------------")
    print(f"Email ({email_input:<25}) : {email_status}")
    print(f"Phone ({phone_input:<25}) : {phone_status}")
    print("--------------------------------------------------------\n")

if __name__ == "__main__":
    main()
