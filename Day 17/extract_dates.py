# ==============================================================================
# Program    : Extract Dates Formatted as DD-MM-YYYY or DD/MM/YYYY
# Objective  : Extract calendar dates with hyphen or slash delimiters using regex.
# Concept    : Date Extraction Pattern (r"\b\d{2}[-/]\d{2}[-/]\d{4}\b")
# Why Used   : Matches 2-digit day, delimiter (- or /), 2-digit month, delimiter, 4-digit year.
# ==============================================================================

import re

sample_text = """
Project kicked off on 15-06-2026 and milestone 1 was completed on 30/07/2026.
Final release planned for 31-12-2026. Invalid date format 2026/05/10.
"""
print("Sample Text:\n", sample_text)

# What is used : Date extraction pattern r"\b\d{2}[-/]\d{2}[-/]\d{4}\b"
# Why it is used: Matches DD-MM-YYYY or DD/MM/YYYY dates cleanly
dates = re.findall(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b", sample_text)

print("\n--- Extracted Dates List ---")
print("Found Dates:", dates)
