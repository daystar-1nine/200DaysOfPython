# ==============================================================================
# Program    : 5 Threads Printing Numbers 1-5 (Task 1)
# Objective  : Create 5 threads that print numbers 1 to 5 concurrently.
# Concept    : Multi-Threaded Concurrent Execution
# Why Used   : Demonstrates creating and joining multiple threads printing sequence outputs.
# ==============================================================================

import threading
import time

def print_numbers(thread_id):
    for i in range(1, 6):
        print(f"[Thread-{thread_id}] Number: {i}")
        time.sleep(0.1)

def main():
    print("=== TASK 1: 5 THREADS PRINTING NUMBERS 1-5 ===")
    threads = []
    
    # What is used : Loop creating 5 threads
    for tid in range(1, 6):
        t = threading.Thread(target=print_numbers, args=(tid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("All 5 threads completed number printing!")

if __name__ == "__main__":
    main()
