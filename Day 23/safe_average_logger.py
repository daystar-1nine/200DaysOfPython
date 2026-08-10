# ==============================================================================
# Program    : Safe Average Function with Exception Logging (Task 3)
# Objective  : Handle empty list division errors gracefully and log exceptions.
# Concept    : Exception Logging & Defensive Error Handling
# Why Used   : Prevents division by zero crashes and logs traceback to file.
# ==============================================================================

import logging
import os

log_file = "average_task3.log"
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def calculate_average(numbers):
    """Calculates average of numbers list. Logs error if list is empty."""
    try:
        avg = sum(numbers) / len(numbers)
        return avg
    except ZeroDivisionError:
        # What is used : logging.exception()
        # Why it is used: Logs error message and appends full traceback details
        logging.exception("Failed to calculate average: Input list is empty!")
        print(f"[ERROR] Cannot calculate average of empty list. Logged to '{log_file}'.")
        return 0.0

def main():
    print("=== TASK 3: SAFE AVERAGE CALCULATOR WITH LOGGING ===")
    
    valid_list = [10, 20, 30, 40]
    empty_list = []

    print(f"Average of {valid_list} = {calculate_average(valid_list)}")
    print(f"Average of {empty_list} = {calculate_average(empty_list)}")

    logging.shutdown()
    if os.path.exists(log_file):
        print(f"\nCaptured Log File '{log_file}' Preview:")
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read())
        os.remove(log_file)

if __name__ == "__main__":
    main()
