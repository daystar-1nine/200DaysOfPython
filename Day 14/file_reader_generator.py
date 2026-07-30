# ==============================================================================
# Program    : File Reader Generator (Bonus Challenge)
# Objective  : Stream a large file line-by-line using a generator to minimize memory usage.
# Concept    : File Streaming Generators & Context Managers
# Why Used   : Reading files via generators processes multi-gigabyte logs without loading entire files into RAM.
# ==============================================================================

import os

# What is used : Generator function 'def read_large_file(filename)'
# Why it is used: Yields file lines one by one lazily
# How it works : Opens file stream; for loop yields line by line with O(1) memory
def read_large_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # What is used : yield statement
            yield line.strip()

def prepare_sample_log(filename):
    """Utility to create a sample log file for testing."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("[INFO] 2026-07-30 10:00:00 - Server Started Successfully.\n")
        f.write("[DEBUG] 2026-07-30 10:00:05 - Connection pool initialized.\n")
        f.write("[WARNING] 2026-07-30 10:01:20 - Memory usage approaching threshold.\n")
        f.write("[INFO] 2026-07-30 10:02:15 - Request processed in 12ms.\n")
        f.write("[ERROR] 2026-07-30 10:03:00 - Database timeout on Query #142.\n")

def main():
    sample_file = "sample_server_log.txt"
    prepare_sample_log(sample_file)

    print(f"=== File Reader Generator (Streaming '{sample_file}') ===")
    
    # What is used : Generator invocation
    line_stream = read_large_file(sample_file)

    line_count = 0
    # What is used : Iterating over generator line stream
    for line in line_stream:
        line_count += 1
        print(f"Line {line_count}: {line}")

    print(f"\nSuccessfully processed {line_count} lines with ultra-low memory usage!")

    # Cleanup sample file
    if os.path.exists(sample_file):
        os.remove(sample_file)

if __name__ == "__main__":
    main()
