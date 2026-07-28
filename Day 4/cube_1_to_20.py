# ==============================================================================
# Program    : Calculate Cube of numbers from 1 to 20
# Objective  : Generate numbers 1-20 and compute their cubic power.
# Concept    : Sequence Generation (for & while loops, range(), exponentiation)
# Why Used   : Compares definite for-loop range iteration vs manual counter while-loop.
# ==============================================================================

print("--- Cubes from 1 to 20 using for loop ---")

# What is used : for loop with range(1, 21)
# Why it is used: range(1, 21) generates integer sequence from 1 up to 20 (stop limit 21 is exclusive)
# How it works : Iterates variable 'i' from 1 to 20 automatically
for i in range(1, 21):
    # What is used : Multiplication arithmetic operator (*)
    # How it works : Multiplies i * i * i to calculate cube (i^3)
    print(i, "Cube =", i * i * i)

print("\n--- Cubes from 1 to 20 using while loop ---")

# What is used : Counter variable initialization 'j = 1'
# Why it is used: Sets initial state starting boundary for while loop
j = 1

# What is used : while loop condition 'j <= 20'
# How it works : Continuously evaluates boolean expression 'j <= 20' before each iteration
while j <= 20:
    print(j, "Cube =", j * j * j)
    # What is used : Augmented assignment operator (j += 1)
    # Why it is used: Increments j by 1 to prevent infinite loop
    j += 1
