# ==============================================================================
# Program    : Concurrency Benchmark Comparison
# Objective  : Compare execution times across Sequential, Threading, Multiprocessing, and Asyncio models.
# Concept    : Paradigm Performance Analysis
# Why Used   : Empirical comparison of I/O wait handling across execution paradigms.
# ==============================================================================

import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def io_task_sync(item):
    time.sleep(0.2)
    return item * 2

async def io_task_async(item):
    await asyncio.sleep(0.2)
    return item * 2

def run_sequential(items):
    t0 = time.time()
    res = [io_task_sync(x) for x in items]
    return time.time() - t0, res

def run_threading(items):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=5) as pool:
        res = list(pool.map(io_task_sync, items))
    return time.time() - t0, res

async def run_asyncio(items):
    t0 = time.time()
    res = await asyncio.gather(*[io_task_async(x) for x in items])
    return time.time() - t0, res

def main():
    print("=== CONCURRENCY BENCHMARK COMPARISON (5 I/O Tasks) ===")
    items = [1, 2, 3, 4, 5]

    seq_time, _ = run_sequential(items)
    print(f"1. Sequential Execution Time   : {seq_time:.4f} s")

    thr_time, _ = run_threading(items)
    print(f"2. ThreadPoolExecutor Time     : {thr_time:.4f} s")

    asy_time, _ = asyncio.run(run_asyncio(items))
    print(f"3. Asyncio Event Loop Time    : {asy_time:.4f} s")

    print("\nConclusion: Threading and Asyncio overlap I/O wait states, achieving ~5x speedup!")

if __name__ == "__main__":
    main()
