# Program: Print Odd Numbers from 1 to 100
# Concept: Loop iteration with condition logic (modulo check for odd numbers)

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
