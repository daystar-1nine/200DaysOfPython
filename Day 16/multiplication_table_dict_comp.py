# ==============================================================================
# Program    : Multiplication Table Dictionary Comprehension
# Objective  : Build a dictionary mapping numbers to their multiplication products.
# Concept    : Dictionary Comprehensions ({key_expr: val_expr for var in iterable})
# Why Used   : Creates key-value mappings in a single declarative pythonic statement.
# ==============================================================================

table_number = 5

# What is used : Dictionary comprehension '{i: table_number * i for i in range(1, 11)}'
# Why it is used: Maps multiplier integer (key) to its calculated product (value)
# How it works : Iterates i through range(1, 11) and sets dict[i] = 5 * i
multiplication_table = {i: table_number * i for i in range(1, 11)}

print(f"=== Multiplication Table for {table_number} (Dict Comprehension) ===")
for multiplier, product in multiplication_table.items():
    print(f"{table_number} x {multiplier:<2} = {product}")
