# Program: Sort a list in ascending and descending order
# Concept: In-place sort() vs sorted() function

numbers = [42, 15, 88, 3, 67, 29, 91]
print("Original List:", numbers)

# Ascending order using sorted() (Returns new list)
ascending_list = sorted(numbers)
print("Ascending Order:", ascending_list)

# Descending order using sorted(reverse=True)
descending_list = sorted(numbers, reverse=True)
print("Descending Order:", descending_list)

# In-place sorting using .sort()
numbers.sort()
print("List after .sort():", numbers)
