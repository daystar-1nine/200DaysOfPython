# ==============================================================================
# Program    : Print Odd Numbers from 1 to 100
# Objective  : Practice and master print odd numbers from 1 to 100 logic.
# Concept    : Loop iteration with condition logic (modulo check for odd numbers)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Odd numbers using for loop ---")
for i in range(1, 100, 2):
    print(i, end=" ")


print("\n--- Odd numbers using while loop ---")
count = 1
while count <= 100:
    if count % 2 != 0:
        print(count, end=" ")
    count += 1
print()
