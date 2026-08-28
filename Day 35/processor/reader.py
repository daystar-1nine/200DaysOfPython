# ==============================================================================
# Program    : Streaming File Reader Generators (reader.py)
# Objective  : Stream lines and CSV dict records from file lazily using yield.
# Concept    : Generator File I/O Streaming
# Why Used   : Prevents loading large files entirely into RAM.
# ==============================================================================

import csv
import os

def read_lines(filename: str):
    """Generator streaming lines from a file one at a time."""
    # What is used : Streaming File Context Manager with yield
    # Why it is used: Opens file and yields lines lazily with O(1) RAM footprint
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            cleaned = line.strip()
            if cleaned:
                yield cleaned

def read_csv_records(filename: str):
    """Generator streaming CSV records as dictionaries lazily."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")

    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield row
