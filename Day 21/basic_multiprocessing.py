# ==============================================================================
# Program    : Basic Multiprocessing CPU Execution
# Objective  : Demonstrate spawning parallel processes to execute CPU-heavy calculations.
# Concept    : multiprocessing.Process & GIL Bypassing
# Why Used   : Spawns independent Python interpreter processes running on distinct CPU cores.
# ==============================================================================

import multiprocessing
import time

def cpu_heavy_sum(limit):
    """Computes sum of squares up to limit (CPU-bound)."""
    return sum(i * i for i in range(limit))

def main():
    print("=== BASIC MULTIPROCESSING DEMO ===")
    limit = 5_000_000

    # What is used : multiprocessing.Process(target=..., args=...)
    # Why it is used: Creates OS processes that execute in true parallel across physical CPU cores
    p1 = multiprocessing.Process(target=cpu_heavy_sum, args=(limit,))
    p2 = multiprocessing.Process(target=cpu_heavy_sum, args=(limit,))

    start_time = time.time()
    p1.start()
    p2.start()

    p1.join()
    p2.join()
    elapsed = time.time() - start_time

    print(f"Executed 2 heavy CPU processes in parallel: {elapsed:.2f} seconds!")

# Mandatory entry point check on Windows to prevent recursive process spawning loops
if __name__ == "__main__":
    main()
