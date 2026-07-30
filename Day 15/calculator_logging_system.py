# ==============================================================================
# Program    : Calculator with Logging Decorator (Bonus Challenge)
# Objective  : Log arithmetic operation lifecycle (Started, Arguments, Result, Completed).
# Concept    : Full Lifecycle Logging Decorator
# Why Used   : Formats structured audit logs for mathematical operations automatically.
# ==============================================================================

import functools

# What is used : Audit Logging Decorator 'def log_calculation(operation_name)'
# Why it is used: Parameterized decorator factory taking operation_name argument
def log_calculation(operation_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(a, b):
            print(f"\n[INFO] {operation_name} Started")
            print(f"[INFO] Inputs: a = {a}, b = {b}")
            
            try:
                result = func(a, b)
                print(f"[INFO] Result = {result}")
                print(f"[INFO] {operation_name} Completed")
                return result
            except ZeroDivisionError as e:
                print(f"[ERROR] {operation_name} Failed: {e}")
                print(f"[INFO] {operation_name} Terminated Abnormally")
                return None
        return wrapper
    return decorator

@log_calculation("Addition")
def add(a, b):
    return a + b

@log_calculation("Subtraction")
def subtract(a, b):
    return a - b

@log_calculation("Multiplication")
def multiply(a, b):
    return a * b

@log_calculation("Division")
def divide(a, b):
    return a / b

def main():
    print("=== Calculator with Audit Logging System ===")
    add(15, 10)
    subtract(50, 20)
    multiply(6, 7)
    divide(100, 4)
    divide(50, 0)

if __name__ == "__main__":
    main()
