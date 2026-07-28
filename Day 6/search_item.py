# ==============================================================================
# Program    : Search for an item in a list
# Objective  : Practice and master search for an item in a list logic.
# Concept    : 'in' operator, index() method, and linear search logic
# Why Used   : Stores ordered, mutable collections of items allowing dynamic modification. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

fruits = ["Apple", "Banana", "Mango", "Orange", "Grapes"]
print("Available Fruits:", fruits)

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
search_term = input("Enter fruit name to search: ").strip()

# Check existence using 'in' keyword
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if search_term in fruits:
    position = fruits.index(search_term)
    print(f"Found '{search_term}' at index {position} (Position {position + 1})")
else:
    print(f"'{search_term}' is NOT available in the list.")
