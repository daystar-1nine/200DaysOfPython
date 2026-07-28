# ==============================================================================
# Program    : Sum of Natural Numbers up to N
# Objective  : Practice and master sum of natural numbers up to n logic.
# Concept    : Accumulation (addition), loop bounds, user input processing
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

num = int(input("Enter number: "))

# Using for loop
sum1 = 0
for i in range(1, num + 1):
    sum1 = sum1 + i

print("Sum using for =", sum1)

# Using while loop
j = 1
sum2 = 0
while j <= num:
    sum2 = sum2 + j
    j += 1

print("Sum using while =", sum2)
