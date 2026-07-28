# ==============================================================================
# Program    : Print Multiplication Table
# Objective  : Practice and master print multiplication table logic.
# Concept    : Basic math operations within loops, loops with start/stop limits
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

num = int(input("Enter Number: "))

print(f"\n--- Multiplication Table for {num} (for loop) ---")
for i in range(0, 13):
    print(num, "x", i, "=", num * i)

print(f"\n--- Multiplication Table for {num} (while loop) ---")
j = 0
while j <= 12:
    print(num, "x", j, "=", num * j)
    j += 1
