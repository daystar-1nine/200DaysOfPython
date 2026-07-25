# Program: Find the average of numbers in a list
# Concept: Sum divided by total count (len())

numbers = [85, 90, 78, 92, 88]
print("Marks List:", numbers)

# Calculate sum and count
total_sum = sum(numbers)
count = len(numbers)

# Calculate average
average = total_sum / count
print(f"Total Marks: {total_sum}")
print(f"Total Subjects: {count}")
print(f"Average Marks: {average:.2f}")
