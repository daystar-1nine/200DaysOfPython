# Program: Calculate Area of a Circle
# Concept: User Input, Type Conversion (int/float), and Arithmetic Operators (** or multiplication)

# Step 1: Take radius input from user and convert it to float (handles both integers and decimals)
radius = float(input("Enter radius of circle: "))

# Constant value of Pi
pie = 3.14159

# Step 2: Calculate Area using formula: Area = π * r²
area = pie * (radius ** 2)

# Step 3: Calculate Circumference using formula: Circumference = 2 * π * r
circumference = 2 * pie * radius

# Step 4: Display the calculated results
print("Area of Circle =", round(area, 2))
print("Circumference of Circle =", round(circumference, 2))