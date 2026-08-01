# ==============================================================================
# Program    : Count and Find Words Starting with 'P'
# Objective  : Extract all words starting with letter 'P' or 'p' using word boundaries.
# Concept    : Word Boundary Matching (r"\b[Pp]\w*")
# Why Used   : \b matches word boundaries, [Pp] matches case-insensitive P, \w* matches remaining word chars.
# ==============================================================================

import re

sample_text = "Python Programming brings Power, Precision, and Performance to Data Pipelines."
print("Sample Text:", sample_text)

# What is used : Word boundary pattern r"\b[Pp]\w*"
# Why it is used: Matches words beginning with uppercase 'P' or lowercase 'p'
# How it works : Finds boundary \b, matches P/p, and captures following word characters
p_words = re.findall(r"\b[Pp]\w*", sample_text)

print("\n--- Extracted Words Starting with 'P' ---")
print("Words found:", p_words)
print("Total count:", len(p_words))
