# ==============================================================================
# Program    : Search for a key in a dictionary
# Objective  : Practice and master search for a key in a dictionary logic.
# Concept    : Checking key existence using 'in' operator and .get()
# Why Used   : Pauses execution to capture interactive user input from standard input. Evaluates conditional expressions to control program execution flow.
# ==============================================================================

phone_book = {
    "Suraj": "9876543210",
    "Rahul": "9123456780",
    "Priya": "9988776655"
}

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
search_name = input("Enter contact name to search: ").strip()

# Method 1: Using 'in' keyword
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if search_name in phone_book:
    print(f"Found! {search_name}'s Number: {phone_book[search_name]}")
else:
    print(f"Contact '{search_name}' not found.")

# Method 2: Using .get() method
number = phone_book.get(search_name, "Contact Not Found")
print(f"Result (.get()): {number}")
