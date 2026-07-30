# ==============================================================================
# Program    : Display Current Date and Time
# Objective  : Format and display current system timestamp using datetime module.
# Concept    : Datetime Object Formatting (datetime.now, strftime)
# Why Used   : datetime module provides complete calendar date and 24h/12h time objects.
# ==============================================================================

# What is used : from datetime import datetime
# Why it is used: Imports datetime class directly into current namespace
from datetime import datetime

# What is used : datetime.now()
# Why it is used: Returns current local date and time system object
# How it works : Queries OS system clock for local timestamp components
now = datetime.now()

print("=== Current System Date & Time ===")
print("Raw Timestamp Object:", now)

# What is used : strftime() method with directive directives
# Why it is used: Formats datetime object into human-readable custom string format
# How it works : %A = Day Name, %d = Day, %B = Month Name, %Y = Year, %I = Hour, %M = Minute, %p = AM/PM
formatted_date = now.strftime("%A, %d %B %Y")
formatted_time = now.strftime("%I:%M:%S %p")

print(f"Formatted Date: {formatted_date}")
print(f"Formatted Time: {formatted_time}")
