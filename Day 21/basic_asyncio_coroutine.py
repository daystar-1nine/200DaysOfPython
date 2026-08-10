# ==============================================================================
# Program    : Basic Asyncio Coroutine (async / await)
# Objective  : Write and execute a basic coroutine using async def and asyncio.run().
# Concept    : Single-Threaded Cooperative Multitasking (asyncio)
# Why Used   : asyncio coroutines yield control via await without blocking the event loop.
# ==============================================================================

import asyncio

# What is used : async def coroutine definition
# Why it is used: Declares an asynchronous function returning a coroutine object
async def async_fetch(service_name):
    print(f"--> [Async] Initiating request to '{service_name}'...")
    
    # What is used : await asyncio.sleep(1)
    # Why it is used: Non-blocking sleep that yields execution back to asyncio event loop
    await asyncio.sleep(1)
    
    print(f"<-- [Async] Received response from '{service_name}'!")
    return f"{service_name}-Data"

async def main():
    print("=== BASIC ASYNCIO COROUTINE DEMO ===")
    result = await async_fetch("AuthServer")
    print("Coroutine Output:", result)

if __name__ == "__main__":
    # What is used : asyncio.run(main())
    # Why it is used: Creates event loop, runs main() coroutine, and closes event loop
    asyncio.run(main())
