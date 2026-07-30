# ==============================================================================
# Program    : Logging Decorator Demonstration
# Objective  : Log function names, arguments, and return values automatically.
# Concept    : Universal Decorators (*args, **kwargs)
# Why Used   : *args and **kwargs allow decorator to accept any function signature dynamically.
# ==============================================================================

import functools

# What is used : Logging Decorator 'def log_execution(func)'
def log_execution(func):
    @functools.wraps(func)
    # What is used : *args and **kwargs in wrapper definition
    # Why it is used: Accepts arbitrary positional and keyword arguments passed to target function
    def wrapper(*args, **kwargs):
        print(f"[LOG INFO] Executing '{func.__name__}' with positional args={args}, kwargs={kwargs}")
        
        # What is used : Invoking target function with arguments
        result = func(*args, **kwargs)
        
        print(f"[LOG SUCCESS] '{func.__name__}' returned result -> {result}\n")
        return result
    return wrapper

@log_execution
def add_numbers(a, b):
    return a + b

@log_execution
def greet_user(name, message="Welcome"):
    return f"{message}, {name}!"

def main():
    print("=== Logging Decorator Demonstration ===")
    add_numbers(15, 30)
    greet_user("Suraj", message="Hello")

if __name__ == "__main__":
    main()
