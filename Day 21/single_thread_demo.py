# ==============================================================================
# Program    : Single Thread Execution Demonstration
# Objective  : Demonstrate basic thread creation, execution, and join synchronization.
# Concept    : Python threading Module (threading.Thread, start, join)
# Why Used   : Spawns an independent thread of execution while main thread waits.
# ==============================================================================

import threading
import time

# Function to run in separate thread
def worker_task(task_name):
    # What is used : time.sleep()
    # Why it is used: Simulates an I/O wait state (e.g. database/network fetch)
    print(f"[Thread] {task_name} started execution...")
    time.sleep(1)
    print(f"[Thread] {task_name} finished execution!")

def main():
    print("=== SINGLE THREAD DEMONSTRATION ===")

    # What is used : threading.Thread(target=..., args=...)
    # Why it is used: Instantiates a thread object bound to target worker_task
    t = threading.Thread(target=worker_task, args=("DataFetcher-1",))

    # What is used : t.start()
    # Why it is used: Initiates thread execution in background
    t.start()

    print("[Main] Waiting for worker thread to complete...")

    # What is used : t.join()
    # Why it is used: Blocks main thread until worker thread finishes
    t.join()

    print("[Main] Worker thread completed. Exiting program.")

if __name__ == "__main__":
    main()
