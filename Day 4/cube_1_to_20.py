# Program: Calculate Cube of numbers from 1 to 20
# Concept: Generating sequences using for loop, while loop, and arithmetic multiplication

print("--- Cubes from 1 to 20 using for loop ---")
for i in range(1, 21):
    print(i, "Cube =", i * i * i)

print("\n--- Cubes from 1 to 20 using while loop ---")
j = 1
while j <= 20:
    print(j, "Cube =", j * j * j)
    j += 1
