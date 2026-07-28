# ==============================================================================
# Program    : Search for an item in a list
# Objective  : Practice and master search for an item in a list logic.
# Concept    : 'in' operator, index() method, and linear search logic
# Why Used   : Stores ordered, mutable collections of items allowing dynamic modification. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

fruits = ["Apple", "Banana", "Mango", "Orange", "Grapes"]
print("Available Fruits:", fruits)

search_term = input("Enter fruit name to search: ").strip()

# Check existence using 'in' keyword
if search_term in fruits:
    position = fruits.index(search_term)
    print(f"Found '{search_term}' at index {position} (Position {position + 1})")
else:
    print(f"'{search_term}' is NOT available in the list.")
