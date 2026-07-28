# ==============================================================================
# Program    : Calculate Area and Perimeter of a Rectangle
# Objective  : Perform standard geometric computations using multiplication and addition.
# Why Used   : Teaches multi-input processing and user-defined formulas.
# ==============================================================================

# Step 1: Input length and width
length = float(input("Enter the length of rectangle: "))
width = float(input("Enter the width of rectangle: "))

# Step 2: Calculate Area (length * width) and Perimeter (2 * (length + width))
area = length * width
perimeter = 2 * (length + width)

# Step 3: Output results
print(f"\nLength    : {length}")
print(f"Width     : {width}")
print(f"Area      : {round(area, 2)}")
print(f"Perimeter : {round(perimeter, 2)}")
