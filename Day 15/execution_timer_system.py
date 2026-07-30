# ==============================================================================
# Program    : Function Execution Timer System (Mini Project)
# Objective  : Benchmark runtime of sorting, searching, and math algorithms using decorators.
# Concept    : Benchmark Decorators with High Precision Timers
# Why Used   : Measures and logs algorithm duration in seconds with 6 decimal places.
# ==============================================================================

import functools
import random
import time

# What is used : Benchmark Decorator 'def measure_time(func)'
def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"\n[Benchmarking] Running '{func.__name__}'...")
        start_time = time.perf_counter()
        
        # Execute target function
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"[Benchmark Result] '{func.__name__}' Finished in {duration:.6f} seconds.")
        return result
    return wrapper

@measure_time
def perform_sorting(data_list):
    """Sorts a list of numbers in ascending order."""
    return sorted(data_list)

@measure_time
def perform_linear_search(data_list, target):
    """Searches for a target number in a list."""
    return target in data_list

@measure_time
def perform_math_calculation(limit):
    """Calculates sum of squares."""
    return sum(i * i for i in range(limit))

def main():
    print("=== Function Execution Timer System ===")
    sample_data = [random.randint(1, 100000) for _ in range(50000)]

    perform_sorting(sample_data)
    perform_linear_search(sample_data, target=99999)
    perform_math_calculation(500000)

if __name__ == "__main__":
    main()
