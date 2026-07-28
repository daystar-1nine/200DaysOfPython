# ==============================================================================
# Program    : Print numbers from 100 down to 1 (Reverse Counting)
# Objective  : Practice and master print numbers from 100 down to 1 (reverse counting) logic.
# Concept    : Negative step limits in range(), decrementing counters in while loops
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Reverse counting using for loop ---")
for i in reversed(range(1,101)):
    print(i, end=" ")
print()

print("\n--- Reverse counting using while loop ---")
j = 100
while j >= 1:
    print(j, end=" ")
    j -= 1
print()
