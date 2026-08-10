# ==============================================================================
# Program    : Multiple Threads Concurrent Downloads
# Objective  : Simulate downloading 3 files concurrently using multiple threads.
# Concept    : Concurrent I/O Operations via Multi-Threading
# Why Used   : Overlaps multiple I/O sleep delays, reducing total execution time.
# ==============================================================================

import threading
import time

def download_file(filename, download_time):
    print(f"--> Starting download: {filename} (takes {download_time}s)")
    # What is used : time.sleep() simulating I/O download wait
    time.sleep(download_time)
    print(f"<-- Finished download: {filename}")

def main():
    print("=== MULTI-THREADED CONCURRENT DOWNLOADS ===")
    files = [("File-A.zip", 2), ("File-B.mp4", 1), ("File-C.pdf", 1.5)]
    
    start_time = time.time()
    threads = []

    # What is used : Spawning multiple threads
    # How it works : Loops through files list, creates thread per file, and calls start()
    for fname, duration in files:
        t = threading.Thread(target=download_file, args=(fname, duration))
        threads.append(t)
        t.start()

    # What is used : t.join() loop
    # How it works : Ensures main thread waits for all download threads to complete
    for t in threads:
        t.join()

    total_elapsed = time.time() - start_time
    print(f"\nAll files downloaded concurrently in {total_elapsed:.2f} seconds!")
    print("(Sequential download would have taken 4.50 seconds).")

if __name__ == "__main__":
    main()
