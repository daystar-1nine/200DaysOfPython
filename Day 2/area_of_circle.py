# ==============================================================================
# Program    : Calculate Area and Circumference of a Circle
# Objective  : Apply mathematical formulas in Python using exponentiation (**).
# Why Used   : Demonstrates float inputs, arithmetic operations, and round() formatting.
# ==============================================================================

import math

# Step 1: Accept radius as float from user
radius = float(input("Enter the radius of the circle: "))

# Step 2: Calculate Area (pi * r^2) and Circumference (2 * pi * r)
# math.pi provides high-precision value for Pi (~3.14159)
area = math.pi * (radius ** 2)
circumference = 2 * math.pi * radius

# Step 3: Display results rounded to 2 decimal places
print(f"\nCircle Radius       : {radius}")
print(f"Area of Circle      : {round(area, 2)}")
print(f"Circumference       : {round(circumference, 2)}")
