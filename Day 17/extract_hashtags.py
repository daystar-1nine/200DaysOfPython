# ==============================================================================
# Program    : Extract Social Media Hashtags
# Objective  : Extract all hashtags starting with '#' followed by word characters.
# Concept    : Hashtag Extraction Pattern (r"#\w+")
# Why Used   : #\w+ matches '#' symbol and contiguous word characters.
# ==============================================================================

import re

post_text = "Learning #Python and #100DaysOfCode! Loving #DataScience and #MachineLearning #200DaysOfPython."
print("Post Content:", post_text)

# What is used : Hashtag regex pattern r"#\w+"
# Why it is used: Extracts all hashtag topics from social media posts
hashtags = re.findall(r"#\w+", post_text)

print("\n--- Extracted Hashtags ---")
print("Hashtags List:", hashtags)
