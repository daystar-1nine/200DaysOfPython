# ==============================================================================
# Program    : Convert list to tuple and tuple to list
# Objective  : Practice and master convert list to tuple and tuple to list logic.
# Concept    : Type casting between tuple() and list() constructors
# Why Used   : Stores ordered, mutable collections of items allowing dynamic modification. Provides an immutable, memory-efficient collection for fixed data.
# ==============================================================================

# Step 1: List to Tuple
programming_languages = ["Python", "Java", "C++", "JavaScript"]
print("Original List:", programming_languages, type(programming_languages))

languages_tuple = tuple(programming_languages)
print("Converted Tuple:", languages_tuple, type(languages_tuple))

# Step 2: Tuple to List
fixed_coordinates = (19.0760, 72.8777)  # Mumbai coordinates
print("\nOriginal Tuple:", fixed_coordinates, type(fixed_coordinates))

coordinates_list = list(fixed_coordinates)
print("Converted List:", coordinates_list, type(coordinates_list))
