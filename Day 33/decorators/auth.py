# ==============================================================================
# Program    : Authorization Decorator (@requires_auth)
# Objective  : Protect sensitive functions by asserting caller authorization role.
# Concept    : Security & Access Control Decorator
# Why Used   : Restricts execution to authorized users with correct role.
# ==============================================================================

from functools import wraps

CURRENT_USER = {"username": "Suraj", "is_authenticated": True, "role": "admin"}

def requires_auth(role: str = "admin"):
    """Decorator checking current user authentication status and role."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not CURRENT_USER.get("is_authenticated"):
                raise PermissionError("Access Denied: User is not authenticated.")
            if CURRENT_USER.get("role") != role:
                raise PermissionError(f"Access Denied: Requires '{role}' role.")
            print(f"[AUTH] User '{CURRENT_USER['username']}' authorized for '{func.__name__}'")
            return func(*args, **kwargs)
        return wrapper
    return decorator
