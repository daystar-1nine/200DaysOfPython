# ==============================================================================
# Program    : Function Execution Timer Decorator
# Objective  : Measure exact execution runtime of functions using time.perf_counter().
# Concept    : Benchmark Decorators (time.perf_counter)
# Why Used   : Provides high-precision time measurement for performance profiling.
# ==============================================================================

import functools
import time

# What is used : Timer Decorator 'def timer_decorator(func)'
def timer_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # What is used : High resolution timer 'time.perf_counter()'
        start_time = time.perf_counter()
        
        # Execute target function
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[TIMER] '{func.__name__}' finished execution in {execution_time:.6f} seconds.")
        return result
    return wrapper

@timer_decorator
def simulate_heavy_computation(n):
    """Simulates CPU heavy computation."""
    total = sum(i * i for i in range(n))
    return total

def main():
    print("=== Timer Decorator Demonstration ===")
    res = simulate_heavy_computation(500000)
    print(f"Computation Result: {res}")

if __name__ == "__main__":
    main()
