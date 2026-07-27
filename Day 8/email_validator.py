# Bonus Challenge: Email Validator
# Features: Validates email format (@ count, domain extension, spaces, username)

def validate_email(email):
    email = email.strip()

    # Rule 1: No spaces allowed
    if " " in email:
        return False, "Email must not contain spaces."

    # Rule 2: Contains exactly one @
    if email.count("@") != 1:
        return False, "Email must contain exactly one '@' symbol."

    username, domain = email.split("@")

    # Rule 3: Username is not empty
    if not username:
        return False, "Username before '@' cannot be empty."

    # Rule 4: Domain must contain a dot and non-empty sub-domain
    if "." not in domain:
        return False, "Domain must contain a valid extension (e.g. .com)."

    # Rule 5: Ends with .com, .in, or .org
    valid_extensions = (".com", ".in", ".org")
    if not domain.endswith(valid_extensions):
        return False, f"Domain extension must end with one of {valid_extensions}."

    return True, "Valid Email ✅"

def main():
    print("====================================")
    print("          EMAIL VALIDATOR           ")
    print("====================================")
    email_input = input("Enter Email: ")

    is_valid, message = validate_email(email_input)

    if is_valid:
        print(f"
{message}")
    else:
        print(f"
Invalid Email ❌")
        print(f"Reason: {message}")

if __name__ == "__main__":
    main()
