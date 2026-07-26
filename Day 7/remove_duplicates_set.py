# Program: Remove duplicates using a set
# Concept: Converting list to set automatically removes duplicates

numbers = [10, 20, 30, 20, 10, 40, 50, 30, 60]
print("Original List with Duplicates:", numbers)

# Convert list to set to remove duplicates
unique_set = set(numbers)
print("Unique Set:", unique_set)

# Convert set back to list if needed
unique_list = list(unique_set)
print("Deduplicated List:", unique_list)
