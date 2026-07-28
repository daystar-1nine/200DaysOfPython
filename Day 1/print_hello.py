# ==============================================================================
# Program    : Hello World Printer
# Objective  : Execute basic print statements and string formatting.
# Concept    : Basic I/O & Formatted String Literals (f-strings)
# Why Used   : Demonstrates built-in print() function and f-string interpolation.
# ==============================================================================

# What is used : Built-in print() function
# Why it is used: Outputs raw string text to standard stdout
# How it works : Takes string literal arguments and writes them to console
print("Hello World!")

# What is used : String variable declaration
# Why it is used: Stores greeting text in memory for reuse
message = "Welcome to 200 Days of Python Challenge"

# What is used : f-string (Formatted String Literal - f"...")
# Why it is used: Evaluates variables directly inside string placeholders {}
# How it works : Replaces {message} with value of message variable at runtime
print(f"Greeting: {message}")
