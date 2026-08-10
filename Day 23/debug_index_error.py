# ==============================================================================
# Program    : Fix Index Out of Bounds & Traceback Explanation (Task 1)
# Objective  : Debug list index out of range loop error using proper bounds checking and logging.
# Concept    : Error Traceback Diagnosis & Safe Iteration
# Why Used   : Range of loop range(4) caused IndexError when array length was 3.
# ==============================================================================

import logging
import os

log_file = "debug_task1.log"
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def buggy_loop_demonstration():
    numbers = [10, 20, 30]
    print("=== TASK 1: DEBUG INDEX ERROR DEMO ===")
    print("Original numbers list:", numbers)

    print("\n1. Fixed Iteration Output:")
    for num in numbers:
        print(f"Number: {num}")

    print("\n2. Demonstrating Safe Index Handling with Logging:")
    for i in range(4):
        try:
            print(f"Index {i} -> Value: {numbers[i]}")
        except IndexError:
            # What is used : logging.exception()
            # Why it is used: Logs exception details and stack traceback to file
            logging.exception("IndexError caught: Index %d out of bounds for list length %d", i, len(numbers))
            print(f"Index {i} -> [OUT OF BOUNDS ERROR - Logged to '{log_file}']")

def main():
    buggy_loop_demonstration()

    logging.shutdown()
    # Cleanup log file after test run
    if os.path.exists(log_file):
        os.remove(log_file)

if __name__ == "__main__":
    main()
