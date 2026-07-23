# Program: Print numbers from 100 down to 1 (Reverse Counting)
# Concept: Negative step limits in range(), decrementing counters in while loops

print("--- Reverse counting using for loop ---")
for i in range(100, 0, -1):
    print(i, end=" ")
print()

print("\n--- Reverse counting using while loop ---")
j = 100
while j >= 1:
    print(j, end=" ")
    j -= 1
print()
