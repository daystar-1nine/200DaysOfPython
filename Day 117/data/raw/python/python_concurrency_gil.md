# Global Interpreter Lock (GIL) and Concurrency in Python

The CPython implementation relies on the Global Interpreter Lock (GIL), a mutual-exclusion lock preventing multiple native operating system threads from executing Python bytecode concurrently. While the GIL protects CPython memory management and reference counting from race conditions, it limits CPU-bound multi-threading to a single physical core.

To achieve true parallelism for CPU-bound workloads, Python programs use multiprocessing, which spawns separate operating system processes with independent memory spaces and separate GIL instances. Conversely, I/O-bound operations (such as network requests or disk reads) release the GIL while waiting for external resources, making threading or cooperative multitasking with asyncio highly effective for concurrent I/O throughput.
