# ==============================================================================
# Program    : Handle Invalid List Index Exception
# Objective  : Safely access list elements using user-provided indices.
# Concept    : Exception Handling (try-except IndexError)
# Why Used   : Referencing a list index outside `0 <= index < len(list)` raises IndexError.
# ==============================================================================

fruits = ["Apple", "Banana", "Mango", "Orange", "Grapes"]
print("Available Fruits List:", fruits)

# What is used : try-except IndexError block
# Why it is used: Catches out-of-range index lookup attempts
# How it works : Traps IndexError when index requested exceeds list boundaries
try:
    index = int(input(f"Enter fruit index (0 to {len(fruits) - 1}): "))
    # What is used : List subscript indexing fruits[index]
    selected_fruit = fruits[index]
    print(f"Selected Fruit at index {index}: {selected_fruit}")

except IndexError:
    # What is used : IndexError exception block
    # Why it is used: Prevents crash when requested index is invalid
    print(f"Index Error: Index out of bounds! Please enter a number between 0 and {len(fruits) - 1}.")

except ValueError:
    # What is used : ValueError fallback handler
    # Why it is used: Handles non-integer input for index
    print("Input Error: Please enter a valid integer index!")
