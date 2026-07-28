# ==============================================================================
# Program    : Temperature Converter (Celsius to Fahrenheit)
# Objective  : Convert temperature units using mathematical conversion formula.
# Why Used   : Demonstrates formula evaluation: F = (C * 9/5) + 32.
# ==============================================================================

# Step 1: Accept temperature in Celsius
celsius = float(input("Enter temperature in Celsius (°C): "))

# Step 2: Apply conversion formula
fahrenheit = (celsius * 9 / 5) + 32

# Step 3: Display converted temperature
print(f"\n{celsius}°C is equal to {round(fahrenheit, 2)}°F")
