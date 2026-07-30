# ==============================================================================
# Program    : Greeting Formatter Decorator
# Objective  : Decorate user greetings by transforming text output to uppercase.
# Concept    : Return Value Transformation Decorators
# Why Used   : Intercepts and formats output strings before returning to caller.
# ==============================================================================

import functools

# What is used : Uppercase Decorator 'def uppercase_decorator(func)'
def uppercase_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Obtain original string result from decorated function
        original_result = func(*args, **kwargs)
        # Transform string to uppercase
        return original_result.upper()
    return wrapper

@uppercase_decorator
def greet(name):
    return f"Welcome to Python masterclass, {name}!"

def main():
    print("=== Greeting Decorator Demonstration ===")
    formatted_greeting = greet("Suraj")
    print("Transformed Output:", formatted_greeting)

if __name__ == "__main__":
    main()
