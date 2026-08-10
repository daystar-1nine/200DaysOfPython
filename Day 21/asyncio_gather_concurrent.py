# ==============================================================================
# Program    : Concurrent Execution via asyncio.gather()
# Objective  : Execute multiple coroutines concurrently using asyncio.gather().
# Concept    : Async Concurrent Task Aggregation
# Why Used   : Runs multiple coroutines concurrently on a single-threaded event loop.
# ==============================================================================

import asyncio
import time

async def download_page(url, delay):
    print(f"Fetching URL: {url} (delay {delay}s)...")
    # Non-blocking async sleep
    await asyncio.sleep(delay)
    print(f"Completed URL: {url}")
    return f"Content of {url}"

async def main():
    print("=== ASYNCIO GATHER CONCURRENT DEMO ===")
    start_time = time.time()

    # What is used : asyncio.gather(*coroutines)
    # Why it is used: Schedules multiple coroutines to run concurrently and collects results in order
    results = await asyncio.gather(
        download_page("https://api.github.com", 1.5),
        download_page("https://python.org", 1.0),
        download_page("https://docs.python.org", 0.5)
    )

    elapsed = time.time() - start_time
    print("\nAll Async Tasks Completed!")
    print(f"Total Execution Time: {elapsed:.2f} seconds.")
    print("Returned Results:", results)

if __name__ == "__main__":
    asyncio.run(main())
