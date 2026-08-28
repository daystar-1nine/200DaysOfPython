# ==============================================================================
# Program    : Logging Decorator (@logger)
# Objective  : Log function entry, positional/keyword arguments, and return values.
# Concept    : Standard Function Decorator Pattern
# Why Used   : Adds function execution logging without altering target function code.
# ==============================================================================

from functools import wraps

def logger(func):
    """Decorator logging function entry, arguments, and return value."""
    # What is used : @wraps(func)
    # Why it is used: Preserves original function's name (__name__) and docstring (__doc__)
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()])
        print(f"[LOG] Calling '{func.__name__}' with args: ({arg_str})")
        result = func(*args, **kwargs)
        print(f"[LOG] '{func.__name__}' returned: {result!r}")
        return result
    return wrapper
