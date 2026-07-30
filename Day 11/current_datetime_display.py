# ==============================================================================
# Program    : Display Current Date and Time
# Objective  : Format and display current system timestamp using datetime module.
# Concept    : Datetime Object Formatting (datetime.now, strftime)
# Why Used   : datetime module provides complete calendar date and time manipulation capabilities.
# ==============================================================================

# What is used : from datetime import datetime
# Why it is used: Imports datetime class directly into current namespace
from datetime import datetime

# What is used : datetime.now()
# Why it is used: Obtains current local system date and time timestamp
# How it works : Queries operating system hardware clock for current year, month, day, hour, min, sec
now = datetime.now()

print("=== Current System Date & Time ===")
print("Raw Timestamp Object:", now)

# What is used : strftime() method with format directives
# Why it is used: Formats datetime object into human-readable custom text format
# How it works : %A = Full Day, %d = Day, %B = Full Month, %Y = Year, %I = Hour (12h), %M = Min, %S = Sec, %p = AM/PM
formatted_date = now.strftime("%A, %d %B %Y")
formatted_time = now.strftime("%I:%M:%S %p")

print(f"Formatted Date: {formatted_date}")
print(f"Formatted Time: {formatted_time}")
