# ==============================================================================
# Program    : Find All Numbers in a String
# Objective  : Extract all numerical digit sequences from unstructured text using re.findall().
# Concept    : Regex Digit Matching (r"\d+")
# Why Used   : r"\d+" matches one or more consecutive digit characters.
# ==============================================================================

import re

sample_text = "Order #1042 was placed on 2026-08-01 for Rs 4500 with 18 percent GST."
print("Sample Text:\n", sample_text)

# What is used : re.findall(pattern, text) with pattern r"\d+"
# Why it is used: Extracts all positive integer sequences from text into a list of strings
# How it works : Scans text left-to-right matching 1+ contiguous digits (\d+)
numbers = re.findall(r"\d+", sample_text)

print("\n--- Extracted Numbers List ---")
print("Numbers found:", numbers)
