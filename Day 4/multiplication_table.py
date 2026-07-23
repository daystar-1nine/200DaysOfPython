# Program: Print Multiplication Table
# Concept: Basic math operations within loops, loops with start/stop limits

num = int(input("Enter Number: "))

print(f"\n--- Multiplication Table for {num} (for loop) ---")
for i in range(0, 13):
    print(num, "x", i, "=", num * i)

print(f"\n--- Multiplication Table for {num} (while loop) ---")
j = 0
while j <= 12:
    print(num, "x", j, "=", num * j)
    j += 1
