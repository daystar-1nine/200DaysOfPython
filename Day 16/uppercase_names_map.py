# ==============================================================================
# Program    : Convert Names to Uppercase Using map()
# Objective  : Transform a list of name strings to uppercase using map().
# Concept    : String Transformations via map()
# Why Used   : Appends str.upper transformation to name elements in a single functional step.
# ==============================================================================

names = ["suraj", "rahul", "priya", "amit", "suresh"]
print("Original Names List:", names)

# What is used : map() with lambda calling upper() string method
# Why it is used: Converts each string to uppercase
# How it works : Executes name.upper() on each element of names
uppercase_names = list(map(lambda name: name.upper(), names))

print("\n--- Uppercase Names Output (via map) ---")
print("Transformed Names:", uppercase_names)
