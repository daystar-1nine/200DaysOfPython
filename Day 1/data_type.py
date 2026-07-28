# ==============================================================================
# Program    : Data Types Demonstration
# Objective  : Explore Python's fundamental built-in data types (int, float, str, bool).
# Why Used   : Demonstrates type inspection using type() and shows how Python 
#              dynamically binds types to variables based on assigned values.
# ==============================================================================

# Step 1: Declare variables of different fundamental data types
age = 20                # Integer (int): Represents whole numbers
height = 5.9            # Floating-point (float): Represents decimal numbers
name = "Suraj Sawant"   # String (str): Sequence of Unicode text characters
is_student = True       # Boolean (bool): Represents truth value (True or False)

# Step 2: Output values and inspect their runtime data types using type()
print("Name       :", name, "| Type:", type(name))
print("Age        :", age, "| Type:", type(age))
print("Height     :", height, "| Type:", type(height))
print("Is Student :", is_student, "| Type:", type(is_student))
