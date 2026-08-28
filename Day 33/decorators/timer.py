# ==============================================================================
# Program    : Timing Decorator (@timer)
# Objective  : Measure execution time of decorated functions using time.perf_counter().
# Concept    : Function Performance Monitoring
# Why Used   : Tracks execution speed of functions and attach execution time metadata.
# ==============================================================================

from functools import wraps
import time

def timer(func):
    """Decorator measuring function execution time in seconds."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # What is used : time.perf_counter()
        # Why it is used: Provides high-resolution monotonic timer ideal for profiling
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[TIMER] '{func.__name__}' executed in {elapsed:.6f} seconds.")
        # Attach execution time metadata attribute to wrapper call
        wrapper.last_execution_time = elapsed
        return result
    wrapper.last_execution_time = 0.0
    return wrapper
