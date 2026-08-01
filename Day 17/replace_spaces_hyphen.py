# ==============================================================================
# Program    : Replace Spaces with Hyphens Using re.sub()
# Objective  : Replace single or multiple whitespace characters with hyphens to form URL slugs.
# Concept    : Regex Substitution (re.sub)
# Why Used   : r"\s+" matches contiguous whitespace chunks and replaces them with a single hyphen.
# ==============================================================================

import re

title_text = "Mastering   Python   Regular   Expressions   In   200   Days"
print("Original Text:", title_text)

# What is used : re.sub(pattern, replacement, text)
# Why it is used: Replaces contiguous spaces (\s+) with single hyphen '-'
# How it works : Scans text for whitespace runs and substitutes '-'
url_slug = re.sub(r"\s+", "-", title_text).lower()

print("\n--- Generated URL Slug ---")
print("URL Slug Output:", url_slug)
