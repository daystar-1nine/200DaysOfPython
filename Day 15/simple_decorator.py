# ==============================================================================
# Program    : Simple Decorator Demonstration
# Objective  : Demonstrate basic decorator pattern wrapping a function call.
# Concept    : Decorators (@decorator syntax & Higher-Order Functions)
# Why Used   : Adds pre-execution and post-execution behavior without modifying target function code.
# ==============================================================================

import functools

# What is used : Decorator function 'def simple_decorator(func)'
# Why it is used: Higher-order function taking a function 'func' as argument
def simple_decorator(func):
    # What is used : @functools.wraps(func)
    # Why it is used: Preserves original function's __name__ and __doc__ metadata
    @functools.wraps(func)
    def wrapper():
        # Pre-execution logic
        print("[Decorator Pre-Action] Preparing to execute function...")
        
        # What is used : Invoking target function 'func()'
        func()
        
        # Post-execution logic
        print("[Decorator Post-Action] Function execution completed!\n")
    
    # What is used : Returning wrapper function object
    return wrapper

# What is used : Syntactic sugar '@simple_decorator'
# How it works : Equivalent to say_hello = simple_decorator(say_hello)
@simple_decorator
def say_hello():
    """Prints a friendly greeting."""
    print("Hello, Python Decorators World!")

def main():
    print("=== Simple Decorator Demonstration ===")
    say_hello()
    print(f"Function Name Preserved: {say_hello.__name__}")

if __name__ == "__main__":
    main()
