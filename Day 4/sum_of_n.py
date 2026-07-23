# Program: Sum of Natural Numbers up to N
# Concept: Accumulation (addition), loop bounds, user input processing

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
