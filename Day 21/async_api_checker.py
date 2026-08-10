# ==============================================================================
# Program    : Async API & Website Status Checker (Challenge Project)
# Objective  : Concurrently check multiple website URLs using asyncio and display HTTP status codes.
# Concept    : Async HTTP Network Probing & Gathering
# Why Used   : Checks multiple web URLs concurrently without blocking the main event loop.
# ==============================================================================

import asyncio
import time
import urllib.request
import urllib.error

# Target URLs to check concurrently
TARGET_URLS = [
    "https://google.com",
    "https://github.com",
    "https://example.com",
    "https://python.org",
    "https://invalid-domain-xyz123.com"
]

async def check_url_status(url):
    """Asynchronously probes a URL and returns status status code / failure."""
    # Running synchronous urllib request inside asyncio executor pool to prevent blocking
    loop = asyncio.get_running_loop()
    try:
        def fetch():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status
        
        status_code = await loop.run_in_executor(None, fetch)
        print(f"{url:<35} [VALID] Status: {status_code}")
        return url, f"[VALID] {status_code}"
    except Exception:
        print(f"{url:<35} [INVALID] Failed / Offline")
        return url, "[INVALID] Failed"

async def main():
    print("==========================================================")
    print("                ASYNC WEBSITE STATUS CHECKER              ")
    print("==========================================================")
    print("Checking website availability concurrently...\n")

    start_time = time.time()

    # What is used : asyncio.gather(*tasks)
    # Why it is used: Runs website checks concurrently on event loop
    tasks = [check_url_status(url) for url in TARGET_URLS]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    print("\n------------------ FINAL CHECK SUMMARY ------------------")
    for url, status in results:
        print(f"{url:<38} -> {status}")
    print(f"\nTotal Checking Time: {elapsed:.2f} seconds")
    print("----------------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
