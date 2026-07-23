# Program: Print Even Numbers from 1 to 100
# Concept: Loop iteration with condition logic (modulo check for even numbers)

print("--- Even numbers using for loop ---")
for i in range(2, 101):
    if i % 2 == 0:
        print(i, end=" ")
print()

print("\n--- Even numbers using while loop ---")
count = 2
while count <= 100:
    if count % 2 == 0:
        print(count, end=" ")
    count += 1
print()
