# ==============================================================================
# Program    : Async Background Task Creation (asyncio.create_task)
# Objective  : Schedule coroutines to run as background tasks on the event loop.
# Concept    : Explicit Task Scheduling (asyncio.create_task)
# Why Used   : Instantly schedules coroutines for background execution while main logic continues.
# ==============================================================================

import asyncio

async def background_logger(message, count):
    for i in range(1, count + 1):
        print(f"[Background Log] {message} - Pulse {i}")
        await asyncio.sleep(0.4)

async def main():
    print("=== ASYNCIO BACKGROUND TASK CREATION ===")

    # What is used : asyncio.create_task(coroutine)
    # Why it is used: Wraps coroutine into a Task object and schedules it on event loop immediately
    task1 = asyncio.create_task(background_logger("Service-Health", 3))
    task2 = asyncio.create_task(background_logger("Metrics-Collector", 3))

    print("[Main Coroutine] Continuing main execution while tasks run in background...")
    
    # Await task completions
    await task1
    await task2

    print("[Main Coroutine] All background tasks finished.")

if __name__ == "__main__":
    asyncio.run(main())
