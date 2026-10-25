# Python Global Interpreter Lock (GIL) and Concurrency

The Global Interpreter Lock (GIL) is a mutex in CPython that prevents multiple native OS threads from executing Python bytecode simultaneously.
The GIL exists primarily because CPython's memory management is not thread-safe without coarse-grained locking.

For CPU-bound tasks, multithreading in Python does not achieve true parallelism across multiple CPU cores due to GIL contention.
In contrast, CPU-bound tasks benefit from `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`, which instantiate separate Python processes with independent GILs and memory spaces.
For I/O-bound operations (such as network requests or disk reads), Python threads release the GIL while waiting on the kernel, enabling high concurrency via threading or `asyncio`.