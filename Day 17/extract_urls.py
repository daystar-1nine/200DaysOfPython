# ==============================================================================
# Program    : Extract URLs from Unstructured Text
# Objective  : Extract web URLs starting with http:// or https:// using regex.
# Concept    : URL Extraction Pattern (r"https?://[^\s]+")
# Why Used   : Matches http, optional 's', '://', and non-whitespace URL characters.
# ==============================================================================

import re

sample_text = """
Visit our website at https://github.com/daystar-1nine/200DaysOfPython for source code.
Check documentation at http://python.org or API docs at https://docs.python.org/3/library/re.html.
"""
print("Sample Text:\n", sample_text)

# What is used : URL extraction pattern r"https?://[^\s]+"
# Why it is used: Extracts web links from multi-line text blocks
urls = re.findall(r"https?://[^\s]+", sample_text)

print("\n--- Extracted Web URLs ---")
for idx, url in enumerate(urls, 1):
    print(f"{idx}. {url}")
