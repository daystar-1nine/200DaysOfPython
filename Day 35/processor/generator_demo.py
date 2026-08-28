# ==============================================================================
# Program    : Generator Generators & Memory Benchmark (generator_demo.py)
# Objective  : Implement Fibonacci, EvenNumbers, Squares generators and memory comparison.
# Concept    : Mathematical Generators & Memory Profiling
# Why Used   : Demonstrates memory difference between list() and generator evaluation.
# ==============================================================================

import sys
import time

def fibonacci_generator(limit: int):
    """Generator producing Fibonacci sequence up to count limit."""
    a, b = 0, 1
    count = 0
    while count < limit:
        yield a
        a, b = b, a + b
        count += 1

def even_numbers_generator(limit: int):
    """Generator producing even numbers up to limit."""
    for n in range(2, limit + 1, 2):
        yield n

def squares_generator(limit: int):
    """Generator producing square numbers up to limit."""
    for n in range(1, limit + 1):
        yield n * n

def benchmark_memory_and_speed(count: int = 1_000_000):
    print(f"\n--- MEMORY BENCHMARK (N = {count:,}) ---")
    
    # Eager List Evaluation
    start_t = time.perf_counter()
    eager_list = [x * x for x in range(count)]
    list_time = time.perf_counter() - start_t
    list_mem = sys.getsizeof(eager_list)

    # Lazy Generator Evaluation
    start_t = time.perf_counter()
    lazy_gen = (x * x for x in range(count))
    gen_time = time.perf_counter() - start_t
    gen_mem = sys.getsizeof(lazy_gen)

    print(f"Eager List : Memory = {list_mem:,} bytes (~{list_mem/(1024*1024):.2f} MB) | Creation Time = {list_time:.4f}s")
    print(f"Lazy Gen   : Memory = {gen_mem:,} bytes (~{gen_mem/1024:.2f} KB) | Creation Time = {gen_time:.6f}s")
    print(f"Memory Reduction Factor: ~{list_mem / gen_mem:,.1f}x smaller RAM footprint!")

if __name__ == "__main__":
    benchmark_memory_and_speed(1_000_000)
