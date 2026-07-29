# ==============================================================================
# Program    : Secure Login System
# Objective  : Authentication system with maximum 3 attempt limit & account lockout.
# Concept    : Exception Handling & Counter State Control
# Why Used   : Prevents brute-force attacks by locking account after 3 failed login attempts.
# ==============================================================================

class AccountLockedError(Exception):
    pass

class EmptyInputError(Exception):
    pass

class InvalidCredentialsError(Exception):
    pass

CORRECT_USERNAME = "admin"
CORRECT_PASSWORD = "Password123"

def login():
    max_attempts = 3
    attempts = 0

    print("=== SECURE LOGIN SYSTEM ===")

    while attempts < max_attempts:
        try:
            username = input("\nEnter Username: ").strip()
            password = input("Enter Password: ").strip()

            # What is used : Empty input check with custom exception
            if not username or not password:
                raise EmptyInputError("Validation Error: Username and password cannot be empty!")

            # What is used : Credential comparison logic
            if username != CORRECT_USERNAME or password != CORRECT_PASSWORD:
                attempts += 1
                remaining = max_attempts - attempts
                if attempts >= max_attempts:
                    # What is used : Raising AccountLockedError on 3rd failure
                    raise AccountLockedError("SECURITY ALERT: Maximum login attempts reached! Your account has been LOCKED.")
                else:
                    raise InvalidCredentialsError(f"Auth Error: Invalid credentials! ({remaining} attempt(s) remaining)")

            # Happy path: credentials match
            print(f"\n[SUCCESS] Welcome, '{username}'! Login authenticated successfully.")
            return

        except EmptyInputError as ee:
            print(ee)
        except InvalidCredentialsError as ie:
            print(ie)
        except AccountLockedError as ae:
            print(f"\n[CRITICAL LOCKOUT] {ae}")
            return

if __name__ == "__main__":
    login()
