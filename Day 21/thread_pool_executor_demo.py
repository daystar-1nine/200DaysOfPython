# ==============================================================================
# Program    : ThreadPoolExecutor High-Level Management
# Objective  : Execute tasks across a pool of reusable worker threads.
# Concept    : concurrent.futures.ThreadPoolExecutor
# Why Used   : Simplifies thread pool lifecycle management and result collection via executor.map().
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor
import time

def process_item(item_id):
    time.sleep(0.5)
    return f"Processed-Item-{item_id}"

def main():
    print("=== THREAD POOL EXECUTOR DEMO ===")
    items = [101, 102, 103, 104, 105, 106]

    start_time = time.time()

    # What is used : ThreadPoolExecutor(max_workers=3) inside context manager
    # Why it is used: Manages 3 worker threads, allocating items to free threads dynamically
    with ThreadPoolExecutor(max_workers=3) as executor:
        # What is used : executor.map(func, iterable)
        # How it works : Maps process_item function over items list concurrently
        results = list(executor.map(process_item, items))

    elapsed = time.time() - start_time
    print("Execution Results:", results)
    print(f"Processed {len(items)} items in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
