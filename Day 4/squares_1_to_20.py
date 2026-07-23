# Program: Print Squares of numbers from 1 to 20
# Concept: Number power operations, range limits, increment operations

print("--- Squares from 1 to 20 using for loop ---")
for i in range(1, 21):
    print(i, "square =", i * i)

print("\n--- Squares from 1 to 20 using while loop ---")
j = 1
while j <= 20:
    print(j, "square =", j * j)
    j += 1
