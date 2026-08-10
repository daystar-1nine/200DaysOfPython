# ==============================================================================
# Program    : Three Async Tasks via asyncio.gather() (Task 6)
# Objective  : Execute 3 async coroutine tasks concurrently using asyncio.gather().
# Concept    : Async Event Loop Task Gathering
# Why Used   : asyncio.gather() runs multiple non-blocking async coroutines concurrently.
# ==============================================================================

import asyncio
import time

async def async_worker(task_id, delay):
    print(f"--> [Async Task-{task_id}] Starting (delay {delay}s)...")
    # What is used : await asyncio.sleep(delay)
    # Why it is used: Non-blocking sleep yielding control back to asyncio event loop
    await asyncio.sleep(delay)
    print(f"<-- [Async Task-{task_id}] Finished!")
    return f"Result-{task_id}"

async def main():
    print("=== TASK 6: THREE ASYNC TASKS VIA ASYNCIO.GATHER ===")
    start_time = time.time()

    # What is used : asyncio.gather(t1, t2, t3)
    # Why it is used: Schedules 3 async tasks to run concurrently on single-threaded event loop
    results = await asyncio.gather(
        async_worker(1, 1.0),
        async_worker(2, 2.0),
        async_worker(3, 1.5)
    )

    elapsed = time.time() - start_time
    print("\nGathered Results:", results)
    print(f"Total Async Execution Time: {elapsed:.2f} seconds (Max delay was 2.0s).")

if __name__ == "__main__":
    asyncio.run(main())
