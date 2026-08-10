# 🐍 Day 21/200 – Masterclass Notes: Concurrency: Threading, Multiprocessing & Asyncio

🎯 **Goal:** Understand advanced Python concurrency mechanisms—**Threading**, **Multiprocessing**, and **Asyncio**—learn when to apply each model, understand race conditions and synchronization locks, and master bypassing Python's GIL for CPU-bound performance.

---

## 📌 Executive Summary & Key Takeaways

- **Concurrency vs. Parallelism:**
  - **Concurrency:** Dealing with multiple things at once (overlapping execution periods on single or multiple cores). Ideal for I/O-bound tasks.
  - **Parallelism:** Doing multiple things simultaneously at the exact same instant across distinct physical CPU cores. Ideal for CPU-bound tasks.
- **Python's Global Interpreter Lock (GIL):**
  - CPython's GIL allows only one thread to execute Python bytecode at a time.
  - Therefore, **Threading** gives concurrency for I/O waiting, but NOT true parallel CPU acceleration.
  - **Multiprocessing** spawns separate OS processes with independent Python interpreters, completely bypassing the GIL for CPU-bound parallel speedups.
- **Asyncio:** Single-threaded cooperative multitasking model built around an **Event Loop**. Uses non-blocking `async def` coroutines and `await` keywords.

---

## 📖 Topic 1: Concurrency Decision Framework

| Task Type | Recommended Mechanism | Primary Bottleneck | Example Use Cases |
|---|---|---|---|
| **I/O-Bound (High IOPS / Network)** | `asyncio` or `threading` | Waiting for Network / Disk / Database | Web Scraping, API Gateway, Chat Server |
| **CPU-Bound (Heavy Math)** | `multiprocessing` | CPU Calculation / Memory Bandwidth | Data Processing, Image Filtering, Machine Learning |
| **Legacy I/O Blocking Code** | `concurrent.futures.ThreadPoolExecutor` | Blocking C Extensions / Sync Drivers | File Batch Operations, Database Queries |

---

## 📖 Topic 2: Threading & Synchronization

### 2.1 Thread Creation & Joining

```python
import threading, time

def fetch_data(source_id):
    print(f"Starting Thread {source_id}")
    time.sleep(1)
    print(f"Finished Thread {source_id}")

threads = []
for i in range(3):
    t = threading.Thread(target=fetch_data, args=(i,))
    threads.append(t)
    t.start()  # Launch thread

for t in threads:
    t.join()   # Wait for thread completion
```

### 2.2 Race Conditions & `threading.Lock()`

```python
import threading

counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    for _ in range(10000):
        # Acquires lock before mutating shared variable
        with lock:
            counter += 1
```

---

## 📖 Topic 3: Multiprocessing (CPU-Bound Parallelism)

### 3.1 Bypassing GIL with Process Pools

```python
from concurrent.futures import ProcessPoolExecutor

def heavy_cpu_task(n):
    return sum(i * i for i in range(n))

# MANDATORY on Windows to prevent recursive process spawning loops
if __name__ == "__main__":
    numbers = [10_000_000, 10_000_000, 10_000_000]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(heavy_cpu_task, numbers))
    print(results)
```

---

## 📖 Topic 4: Asyncio Coroutines (`async` / `await`)

### 4.1 Non-Blocking Event Loop Execution

```python
import asyncio

async def fetch_api(service_name, delay):
    print(f"Fetching {service_name}...")
    await asyncio.sleep(delay)  # Yields control back to event loop
    print(f"Received {service_name}")
    return f"{service_name} Response"

async def main():
    # Runs tasks concurrently on single-threaded event loop
    results = await asyncio.gather(
        fetch_api("AuthService", 2),
        fetch_api("PaymentService", 1),
        fetch_api("InventoryService", 1.5)
    )
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## ⚡ Master Cheat Sheet

```python
# Concurrency Master Cheat Sheet

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# 1. ThreadPoolExecutor (I/O)
with ThreadPoolExecutor(max_workers=4) as pool:
    io_results = list(pool.map(lambda url: f"Fetched {url}", ["url1", "url2"]))

# 2. ProcessPoolExecutor (CPU)
# Required inside if __name__ == "__main__":
with ProcessPoolExecutor() as pool:
    cpu_results = list(pool.map(lambda x: x**2, [1, 2, 3]))

# 3. Asyncio Gather (Single-Threaded I/O)
async def async_main():
    res = await asyncio.gather(asyncio.sleep(1), asyncio.sleep(1))
# asyncio.run(async_main())
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Omitting `if __name__ == "__main__":` on Windows for Multiprocessing:**
   - ❌ `Process()` executed at top level of script causes endless recursive process spawning loops (`RuntimeError` / system crash).
   - ✅ Always wrap process spawning inside `if __name__ == "__main__":`.

2. **Blocking the Asyncio Event Loop with Synchronous Calls:**
   - ❌ Calling `time.sleep()` or synchronous requests inside `async def` blocks all coroutines.
   - ✅ Use `await asyncio.sleep()` or non-blocking async drivers (e.g. `httpx`, `aiohttp`).

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: Why doesn't multithreading accelerate CPU-bound tasks in CPython?
**Answer:** Due to the Global Interpreter Lock (GIL), CPython permits only one OS thread to execute Python bytecode at a time. Threads yield the GIL during I/O wait states, but for CPU-bound code, threads contend for the lock sequentially.

### Q2: What is the main operational difference between Threading and Asyncio?
**Answer:** Threading uses OS-managed preemptive context switching (the OS decides when to switch threads). Asyncio uses single-threaded cooperative multitasking (coroutines explicitly yield execution control using `await`).

---

## 📝 Recap Checklist
- [x] Differentiated between Sequential, Concurrent, and Parallel execution models.
- [x] Understood the CPython GIL and how it influences Threading vs Multiprocessing choices.
- [x] Used `threading.Thread` and `threading.Lock()` to prevent race conditions.
- [x] Used `concurrent.futures.ThreadPoolExecutor` and `ProcessPoolExecutor`.
- [x] Wrote non-blocking coroutines using `async def` and `await asyncio.gather()`.
- [x] Built a Concurrent File Downloader, Multiprocess Image Processor, and Async Web Scraper Simulator.
