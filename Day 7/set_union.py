# ==============================================================================
# Program    : Find union of two sets
# Objective  : Practice and master find union of two sets logic.
# Concept    : Union combines all unique elements from both sets using union() or | operator
# Why Used   : Automatically eliminates duplicate elements and enables set theory operations.
# ==============================================================================

set_a = {"Python", "Java", "C++"}
set_b = {"JavaScript", "Python", "HTML"}

print("Set A:", set_a)
print("Set B:", set_b)

# Method 1: Using union() method
all_languages = set_a.union(set_b)
print("Union (using .union()):", all_languages)

# Method 2: Using | operator
all_languages_operator = set_a | set_b
print("Union (using | operator):", all_languages_operator)
