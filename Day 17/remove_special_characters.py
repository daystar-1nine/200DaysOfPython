# ==============================================================================
# Program    : Remove Special Characters Using Regex
# Objective  : Strip out punctuation and symbols, preserving letters, numbers, and spaces.
# Concept    : Character Class Negation (r"[^\w\s]")
# Why Used   : [^\w\s] matches any character that is NOT a word character or whitespace.
# ==============================================================================

import re

raw_text = "Hello!! Welcome to Python, Day #17... Let's clean @this $text & data! :-)"
print("Raw Dirty Text:", raw_text)

# What is used : Negated character set r"[^\w\s]" inside re.sub()
# Why it is used: Replaces punctuation and special symbols with empty string
# How it works : Preserves alphanumeric characters and spaces, removing everything else
clean_text = re.sub(r"[^\w\s]", "", raw_text)

print("\n--- Cleaned Text ---")
print("Sanitized Text:", clean_text)
