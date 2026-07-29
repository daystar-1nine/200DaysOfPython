# ==============================================================================
# Program    : Handle Missing Dictionary Key Exception
# Objective  : Access dictionary values safely and handle missing key lookups.
# Concept    : Exception Handling (try-except KeyError) vs dict.get()
# Why Used   : Direct bracket lookup `dict[key]` raises KeyError if key is missing.
# ==============================================================================

student_scores = {
    "Suraj": 95,
    "Rahul": 88,
    "Amit": 78
}

print("Student Database Keys:", list(student_scores.keys()))
search_name = input("Enter student name to search score: ").strip()

# What is used : try-except KeyError block
# Why it is used: Demonstrates catching KeyError on direct bracket dictionary access
# How it works : Traps KeyError when search_name is not present in dictionary keys
try:
    score = student_scores[search_name]
    print(f"Score for {search_name}: {score}")

except KeyError:
    # What is used : KeyError exception handler
    # Why it is used: Informs user gracefully that student key does not exist
    print(f"Key Error: Student '{search_name}' was not found in the record system!")
