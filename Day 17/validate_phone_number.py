# ==============================================================================
# Program    : Validate Indian Phone Numbers Using Regex
# Objective  : Verify whether input phone numbers follow 10-digit Indian mobile format.
# Concept    : Anchor & Quantifier Matching (r"^[6-9]\d{9}$")
# Why Used   : Ensures number starts with digit 6, 7, 8, or 9 followed by exactly 9 digits (total 10 digits).
# ==============================================================================

import re

def validate_indian_phone(phone_str):
    # What is used : Indian mobile regex pattern r"^[6-9]\d{9}$"
    # Why it is used: ^ anchors start, [6-9] restricts first digit, \d{9} mandates 9 digits, $ anchors end
    pattern = r"^[6-9]\d{9}$"
    return bool(re.match(pattern, phone_str.strip()))

def main():
    test_numbers = ["9876543210", "8123456789", "5123456789", "98765432", "98765432109", "7000111222"]
    print("=== Phone Number Validation ===")
    for phone in test_numbers:
        is_valid = validate_indian_phone(phone)
        status = "[VALID]" if is_valid else "[INVALID]"
        print(f"Phone: {phone:<15} -> {status}")

if __name__ == "__main__":
    main()
