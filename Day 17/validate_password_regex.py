# ==============================================================================
# Program    : Validate Password Using Regex Lookaheads
# Objective  : Verify password policy compliance using positive lookaheads.
# Concept    : Regex Lookahead Assertions (?=.*[A-Z])
# Why Used   : Checks multiple independent rules (uppercase, lowercase, digit, special symbol, length >= 8).
# ==============================================================================

import re

def is_strong_password(password):
    # What is used : Positive Lookahead Pattern r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    # Why it is used: Asserts presence of lowercase, uppercase, digit, and special char without consuming string
    pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
    return bool(re.match(pattern, password))

def main():
    passwords = ["Pass123!", "weakpass", "NOSPECIAL123", "Short1!", "Secure@Pass2026"]
    print("=== Password Security Policy Check ===")
    for pwd in passwords:
        valid = is_strong_password(pwd)
        status = "[STRONG]" if valid else "[WEAK]"
        print(f"Password: {pwd:<18} -> {status}")

if __name__ == "__main__":
    main()
