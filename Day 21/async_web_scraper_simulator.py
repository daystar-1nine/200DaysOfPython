# ==============================================================================
# Program    : Async Web Scraper Simulator (Bonus Challenge)
# Objective  : Simulate fetching and parsing 5 web page URLs concurrently using asyncio.
# Concept    : Async Cooperative Multitasking with Timeout Handling
# Why Used   : asyncio.gather() and asyncio.wait_for() execute non-blocking web scrapes.
# ==============================================================================

import asyncio
import time

URL_LIST = [
    ("https://news.ycombinator.com", 1.2),
    ("https://github.com/trending", 0.8),
    ("https://pypi.org/search", 1.0),
    ("https://docs.python.org/3/", 0.5),
    ("https://stackoverflow.com/questions", 1.5)
]

async def scrape_url(url, delay):
    print(f"[Scraper] Initiating GET -> {url} (Latency {delay}s)")
    try:
        # Non-blocking async sleep simulating HTTP fetch
        await asyncio.sleep(delay)
        print(f"[Scraper] Successfully parsed DOM -> {url}")
        return {"url": url, "status": 200, "bytes_parsed": len(url) * 45}
    except asyncio.CancelledError:
        print(f"[Scraper] Cancelled GET -> {url}")
        raise

async def main():
    print("==========================================================")
    print("             ASYNC WEB SCRAPER SIMULATOR                  ")
    print("==========================================================")

    start_time = time.time()

    # What is used : List comprehension creating coroutine tasks
    tasks = [scrape_url(url, delay) for url, delay in URL_LIST]

    # What is used : asyncio.gather(*tasks)
    # Why it is used: Concurrent execution of all 5 web scrapes on single-threaded event loop
    scraped_data = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    print("\n---------------- SCRAPING REPORT SUMMARY ----------------")
    print(f"Total URLs Scraped  : {len(scraped_data)}")
    print(f"Total Async Time    : {elapsed:.2f} seconds")
    for item in scraped_data:
        print(f"URL: {item['url']:<35} | Status: {item['status']} | Bytes: {item['bytes_parsed']}")
    print("----------------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
