# ==============================================================================
# Program    : Data Types Demonstration
# Objective  : Explore Python's fundamental built-in data types (int, float, str, bool).
# Concept    : Dynamic Typing & Type Inspection
# Why Used   : Demonstrates type inspection using type() and dynamic binding.
# ==============================================================================

# What is used : Integer (int) data type
# Why it is used: Stores whole numbers without decimal points
# How it works : Binds name 'age' to integer object 20 in memory
age = 20

# What is used : Floating-point (float) data type
# Why it is used: Stores real numbers with decimal precision
height = 5.9

# What is used : String (str) data type
# Why it is used: Stores text sequence of characters
name = "Suraj Sawant"

# What is used : Boolean (bool) data type
# Why it is used: Stores binary state values (True or False)
is_student = True

# What is used : Built-in type() function inside print()
# Why it is used: Returns data type class of variable
# How it works : Inspects internal __class__ attribute of object
print("Name       :", name, "| Type:", type(name))
print("Age        :", age, "| Type:", type(age))
print("Height     :", height, "| Type:", type(height))
print("Is Student :", is_student, "| Type:", type(is_student))
