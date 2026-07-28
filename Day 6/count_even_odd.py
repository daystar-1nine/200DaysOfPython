# ==============================================================================
# Program    : Count even and odd numbers in a list
# Objective  : Practice and master count even and odd numbers in a list logic.
# Concept    : Loop iteration combined with modulo operator (%)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

numbers = [12, 7, 19, 24, 33, 40, 55, 62, 81, 90]
print("Original List:", numbers)

even_count = 0
odd_count = 0

even_numbers = []
odd_numbers = []

for num in numbers:
    if num % 2 == 0:
        even_count += 1
        even_numbers.append(num)
    else:
        odd_count += 1
        odd_numbers.append(num)

print(f"Even Numbers ({even_count}): {even_numbers}")
print(f"Odd Numbers ({odd_count}): {odd_numbers}")
