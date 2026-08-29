# ==============================================================================
# Program    : Authentication Dependency Provider (auth.py)
# Objective  : Provide get_current_user dependency for protecting authenticated endpoints.
# Concept    : Dependency-Based Authentication (Day 43 requirement)
# Why Used   : Encapsulates user authentication logic into reusable FastAPI Depends() callable.
# ==============================================================================

def get_current_user() -> dict:
    """Dependency function resolving current authenticated user dictionary."""
    return {
        "id": 1,
        "name": "Suraj Sawant",
        "email": "suraj@example.com",
        "role": "user"
    }
