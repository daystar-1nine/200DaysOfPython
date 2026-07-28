# ==============================================================================
# Program    : Calculate the area of a circle using a function
# Objective  : Practice and master calculate the area of a circle using a function logic.
# Concept    : Mathematical formulas inside functions
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

def calculate_area(radius):
    """Returns the area of a circle for a given radius"""
    pi = 3.14159
    return pi * (radius ** 2)

# Test the function
r = float(input("Enter the radius of the circle: "))
print(f"Area of the circle is: {round(calculate_area(r), 2)}")
