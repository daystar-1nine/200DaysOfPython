# ==============================================================================
# Program    : Race Condition & Thread Synchronization (Lock)
# Objective  : Demonstrate shared resource race conditions and fix using threading.Lock().
# Concept    : Thread Synchronization & Mutual Exclusion (threading.Lock)
# Why Used   : threading.Lock() ensures atomic access to critical sections.
# ==============================================================================

import threading

# Shared state counter
shared_counter = 0
# What is used : threading.Lock()
# Why it is used: Provides mutual exclusion lock to protect shared_counter
lock = threading.Lock()

def synchronized_increment():
    global shared_counter
    for _ in range(10000):
        # What is used : with lock: context manager
        # How it works : Acquires lock before incrementing, releases lock automatically afterwards
        with lock:
            shared_counter += 1

def main():
    global shared_counter
    shared_counter = 0
    print("=== THREAD SYNCHRONIZATION (LOCK) DEMO ===")

    threads = []
    for _ in range(5):
        t = threading.Thread(target=synchronized_increment)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Expected Counter Value : 50000")
    print(f"Actual Counter Value   : {shared_counter}")

if __name__ == "__main__":
    main()
