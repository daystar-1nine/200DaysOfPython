# ==============================================================================
# Program    : Parameterized Retry Decorator (@retry)
# Objective  : Automatically retry failing functions up to max_attempts with delay.
# Concept    : Parameterized Decorator Factory
# Why Used   : Provides resilient execution for unstable remote network calls.
# ==============================================================================

from functools import wraps
import time

def retry(max_attempts: int = 3, delay: float = 0.05):
    """Decorator factory returning retry decorator with attempts limit and delay."""
    # What is used : Decorator Factory Pattern (Three nested functions)
    # Why it is used: Outer function captures parameters (max_attempts, delay) for closure
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"[RETRY] Attempt {attempt}/{max_attempts} for '{func.__name__}' failed: {e}")
                    if attempt < max_attempts and delay > 0:
                        time.sleep(delay)
            print(f"[RETRY] All {max_attempts} attempts failed for '{func.__name__}'. Raising exception.")
            raise last_exception
        return wrapper
    return decorator
