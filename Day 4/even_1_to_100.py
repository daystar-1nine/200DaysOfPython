# ==============================================================================
# Program    : Print Even Numbers from 1 to 100
# Objective  : Filter and display even numbers in range 1-100.
# Concept    : Modulus Operator (%) & Loop Iteration
# Why Used   : Modulus operator (% 2 == 0) tests divisibility by 2.
# ==============================================================================

print("--- Even numbers using for loop ---")

# What is used : for loop with range(2, 101)
# How it works : Cycles 'i' through numbers 2 to 100
for i in range(2, 101):
    # What is used : Modulus operator (%) inside if condition
    # Why it is used: 'i % 2 == 0' checks if remainder of i divided by 2 is zero (even number)
    if i % 2 == 0:
        # What is used : end=" " parameter in print()
        # Why it is used: Replaces default newline with space to print output on single line
        print(i, end=" ")
print()

print("\n--- Even numbers using while loop ---")

# What is used : Loop counter state 'count = 2'
count = 2

# What is used : while loop condition 'count <= 100'
while count <= 100:
    if count % 2 == 0:
        print(count, end=" ")
    count += 1
print()
