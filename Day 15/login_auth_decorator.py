# ==============================================================================
# Program    : Login Authentication Decorator (Challenge Project)
# Objective  : Protect sensitive system functions using @login_required decorator.
# Concept    : Authorization & Security Decorators
# Why Used   : Verifies authentication state before permitting access to protected features.
# ==============================================================================

import functools

# Simulated global user session state
user_session = {
    "is_logged_in": False,
    "username": None
}

# What is used : Authorization Decorator 'def login_required(func)'
# Why it is used: Checks if user_session["is_logged_in"] is True before executing target function
# How it works : Blocks execution if unauthenticated; executes original function if authenticated
def login_required(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not user_session["is_logged_in"]:
            print(f"[AUTH DENIED] Access Denied to '{func.__name__}'! Please Login First.")
            return None
        print(f"[AUTH GRANTED] Access Granted for user '{user_session['username']}' -> Executing '{func.__name__}'")
        return func(*args, **kwargs)
    return wrapper

@login_required
def view_dashboard():
    print(">>> Displaying Confidential User Dashboard Data <<<")

@login_required
def view_financial_report():
    print(">>> Displaying Q3 Financial Earnings Report <<<")

def login_user(username):
    user_session["is_logged_in"] = True
    user_session["username"] = username
    print(f"\n[LOGIN SUCCESS] User '{username}' logged in successfully!")

def logout_user():
    print(f"\n[LOGOUT] User '{user_session['username']}' logged out.")
    user_session["is_logged_in"] = False
    user_session["username"] = None

def main():
    print("=== Login Authentication System ===")

    print("\n--- Phase 1: Attempting access while logged OUT ---")
    view_dashboard()
    view_financial_report()

    print("\n--- Phase 2: Logging in ---")
    login_user("Suraj Sawant")

    print("\n--- Phase 3: Attempting access while logged IN ---")
    view_dashboard()
    view_financial_report()

    logout_user()

if __name__ == "__main__":
    main()
