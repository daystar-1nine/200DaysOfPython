# ==============================================================================
# Program    : Print Even Numbers from 1 to 100
# Objective  : Practice and master print even numbers from 1 to 100 logic.
# Concept    : Loop iteration with condition logic (modulo check for even numbers)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

print("--- Even numbers using for loop ---")
for i in range(2, 101, 2):
        print(i, end=" ")
print()

print("\n--- Even numbers using while loop ---")
count = 2
while count <= 100:
    if count % 2 == 0:
        print(count, end=" ")
    count += 1
print()
