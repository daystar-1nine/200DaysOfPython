# ==============================================================================
# Program    : Print Squares of numbers from 1 to 20
# Objective  : Practice and master print squares of numbers from 1 to 20 logic.
# Concept    : Number power operations, range limits, increment operations
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Squares from 1 to 20 using for loop ---")
for i in range(1, 21):
    print(i, "square =", i * i)

print("\n--- Squares from 1 to 20 using while loop ---")
j = 1
while j <= 20:
    print(j, "square =", j * j)
    j += 1
