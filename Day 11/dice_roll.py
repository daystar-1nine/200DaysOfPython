# ==============================================================================
# Program    : Roll a Single Dice
# Objective  : Simulate rolling a 6-sided die.
# Concept    : Integer Randomization (random.randint)
# Why Used   : random.randint(1, 6) simulates 6 discrete uniform face values.
# ==============================================================================

# What is used : import random
import random

# What is used : random.randint(1, 6)
# Why it is used: Returns random integer between 1 and 6 inclusive
# How it works : Selects integer face value from set {1, 2, 3, 4, 5, 6}
dice_face = random.randint(1, 6)

print("Rolling 6-sided dice...")
print(f"Outcome: [Dice Face: {dice_face}]")
