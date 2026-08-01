# ==============================================================================
# Program    : Extract Email Addresses from Text
# Objective  : Parse unstructured text and extract all email addresses using regex.
# Concept    : Email Extraction Pattern (r"[\w\.-]+@[\w\.-]+\.\w+")
# Why Used   : Matches standard email format usernames, domain names, and TLD extensions.
# ==============================================================================

import re

sample_text = """
Contact our support team at support@company.com or sales@company.co.in.
For personal queries, reach out to suraj.sawant19@gmail.com or admin_test@dev.org.
"""
print("Sample Input Text:", sample_text)

# What is used : Email extraction regex pattern r"[\w\.-]+@[\w\.-]+\.\w+"
# Why it is used: Identifies valid email addresses inside multi-line text
# How it works : Matches 1+ word/dot/dash chars before '@', domain after '@', and TLD extension after '.'
email_pattern = r"[\w\.-]+@[\w\.-]+\.\w+"
extracted_emails = re.findall(email_pattern, sample_text)

print("--- Extracted Email Addresses ---")
for idx, email in enumerate(extracted_emails, 1):
    print(f"{idx}. {email}")
