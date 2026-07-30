# ==============================================================================
# Program    : Generate Random Number
# Objective  : Produce pseudo-random integers and floats within specified ranges.
# Concept    : Standard Library Import (random module)
# Why Used   : random module provides PRNG algorithms like Mersenne Twister for simulations.
# ==============================================================================

# What is used : import random
# Why it is used: Loads pseudo-random number generator utilities
import random

print("=== Random Number Generator ===")

# What is used : random.randint(1, 100)
# Why it is used: Generates a random integer inclusive of both endpoints (1 and 100)
# How it works : Returns pseudo-random integer N such that 1 <= N <= 100
rand_int = random.randint(1, 100)
print(f"Random Integer (1 to 100) : {rand_int}")

# What is used : random.uniform(10.0, 50.0)
# Why it is used: Generates a random floating-point decimal number between 10.0 and 50.0
rand_float = random.uniform(10.0, 50.0)
print(f"Random Float (10.0 to 50.0): {rand_float:.2f}")
