# ==============================================================================
# Program    : 5 Threads Simulating Downloads (Task 2)
# Objective  : Create 5 threads that simulate file downloads using time.sleep().
# Concept    : Multi-Threaded Download Simulation
# Why Used   : Overlaps time.sleep() delays across 5 concurrent download threads.
# ==============================================================================

import threading
import time

def simulate_file_download(file_id):
    print(f"--> [Thread-{file_id}] Starting download of File_{file_id}.pkg...")
    # What is used : time.sleep()
    # Why it is used: Simulates network latency wait state
    time.sleep(1.5)
    print(f"<-- [Thread-{file_id}] Successfully downloaded File_{file_id}.pkg!")

def main():
    print("=== TASK 2: 5 THREADS SIMULATING FILE DOWNLOADS ===")
    start_time = time.time()
    threads = []

    for fid in range(1, 6):
        t = threading.Thread(target=simulate_file_download, args=(fid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    total_time = time.time() - start_time
    print(f"\nAll 5 file downloads completed concurrently in {total_time:.2f} seconds!")

if __name__ == "__main__":
    main()
