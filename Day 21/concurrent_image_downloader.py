# ==============================================================================
# Program    : Concurrent Image Downloader & Benchmark (Bonus Challenge)
# Objective  : Download 5 images sequentially vs concurrently and compare execution time.
# Concept    : Sequential vs Multi-Threaded Concurrent Downloading Benchmark
# Why Used   : Empirical comparison demonstrating 4-5x speedup from thread-level concurrency.
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor
import os
import time

IMAGE_URLS = [
    ("Image-1.png", 0.5),
    ("Image-2.png", 0.6),
    ("Image-3.png", 0.4),
    ("Image-4.png", 0.5),
    ("Image-5.png", 0.7)
]

def simulate_download(img_data):
    name, delay = img_data
    time.sleep(delay)
    return name

def run_sequential_downloads():
    print("\n--- 1. Running Sequential Image Downloads ---")
    start = time.time()
    results = []
    for img in IMAGE_URLS:
        res = simulate_download(img)
        print(f"Downloaded: {res}")
        results.append(res)
    elapsed = time.time() - start
    return elapsed

def run_concurrent_downloads():
    print("\n--- 2. Running Concurrent Image Downloads (ThreadPool) ---")
    start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(simulate_download, IMAGE_URLS))
        for res in results:
            print(f"Downloaded: {res}")
    elapsed = time.time() - start
    return elapsed

def main():
    print("==========================================================")
    print("      CONCURRENT VS SEQUENTIAL IMAGE DOWNLOADER BENCHMARK ")
    print("==========================================================")

    seq_time = run_sequential_downloads()
    con_time = run_concurrent_downloads()

    print("\n==========================================================")
    print("                   BENCHMARK RESULTS                      ")
    print("==========================================================")
    print(f"Sequential Execution Time : {seq_time:.2f} seconds")
    print(f"Concurrent Execution Time : {con_time:.2f} seconds")
    speedup = seq_time / con_time if con_time > 0 else 1.0
    print(f"Concurrency Speedup Factor : {speedup:.2f}x Faster! [FAST]")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
