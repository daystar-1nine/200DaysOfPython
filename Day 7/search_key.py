# Program: Search for a key in a dictionary
# Concept: Checking key existence using 'in' operator and .get()

phone_book = {
    "Suraj": "9876543210",
    "Rahul": "9123456780",
    "Priya": "9988776655"
}

search_name = input("Enter contact name to search: ").strip()

# Method 1: Using 'in' keyword
if search_name in phone_book:
    print(f"Found! {search_name}'s Number: {phone_book[search_name]}")
else:
    print(f"Contact '{search_name}' not found.")

# Method 2: Using .get() method
number = phone_book.get(search_name, "Contact Not Found")
print(f"Result (.get()): {number}")
