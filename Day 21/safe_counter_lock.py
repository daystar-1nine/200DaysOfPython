# ==============================================================================
# Program    : Shared Counter Safe Increment with Lock (Task 3)
# Objective  : Create a shared counter and safely increment it across multiple threads using Lock.
# Concept    : Mutual Exclusion with threading.Lock()
# Why Used   : Prevents race conditions when multiple threads mutate shared integer state.
# ==============================================================================

import threading

# Shared mutable state
shared_counter = 0

# What is used : threading.Lock()
# Why it is used: Ensures only one thread can modify shared_counter at a time
counter_lock = threading.Lock()

def safe_increment(iterations):
    global shared_counter
    for _ in range(iterations):
        # What is used : with counter_lock:
        # How it works : Acquires lock, executes atomic increment, releases lock
        with counter_lock:
            shared_counter += 1

def main():
    global shared_counter
    shared_counter = 0
    print("=== TASK 3: SHARED COUNTER WITH THREADING LOCK ===")
    
    threads = []
    iterations_per_thread = 10000
    num_threads = 5

    for _ in range(num_threads):
        t = threading.Thread(target=safe_increment, args=(iterations_per_thread,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    expected = num_threads * iterations_per_thread
    print(f"Target Counter Value : {expected}")
    print(f"Actual Counter Value : {shared_counter}")
    print(f"Race Condition Fixed : {shared_counter == expected}")

if __name__ == "__main__":
    main()
