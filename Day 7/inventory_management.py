# ==============================================================================
# Program    : Bonus Challenge: Inventory Management System
# Objective  : Practice and master bonus challenge: inventory management system logic.
# Concept    : Product, Quantity, Price, Calculate total inventory value
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

inventory = {
    "Laptop": {"quantity": 5, "price": 65000},
    "Mouse": {"quantity": 20, "price": 800},
    "Keyboard": {"quantity": 15, "price": 1200}
}


# What is used : Function definition 'def display_inventory'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def display_inventory():
    print("\n=======================================================")
    print("              INVENTORY MANAGEMENT SYSTEM")
    print("=======================================================")
    print(f"{'Product':<15} {'Quantity':<10} {'Price (Rs.)':<12} {'Total Value (Rs.)':<15}")
    print("-------------------------------------------------------")
    
    grand_total = 0

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for product, details in inventory.items():
        qty = details["quantity"]
        price = details["price"]
        total = qty * price
        grand_total += total
        print(f"{product:<15} {qty:<10} {price:<12} {total:<15}")
    
    print("-------------------------------------------------------")
    print(f"GRAND TOTAL INVENTORY VALUE: Rs.{grand_total:,}")
    print("=======================================================")


# What is used : Function definition 'def add_or_update_product'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def add_or_update_product():
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    product = input("Enter Product Name: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not product:
        print("Product name cannot be empty.")
        return
    try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        qty = int(input("Enter Quantity: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        price = float(input("Enter Price per unit (Rs.): "))
        
        inventory[product] = {"quantity": qty, "price": price}
        print(f"Product '{product}' added/updated successfully!")
    except ValueError:
        print("Invalid input! Quantity and price must be numbers.")


# What is used : Function definition 'def remove_product'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def remove_product():
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    product = input("Enter Product Name to remove: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if product in inventory:
        del inventory[product]
        print(f"Product '{product}' removed from inventory.")
    else:
        print("Product not found.")


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n--- Inventory Menu ---")
        print("1. Display Inventory & Total Value")
        print("2. Add / Update Product")
        print("3. Remove Product")
        print("4. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select an option (1-4): ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
            display_inventory()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
            add_or_update_product()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
            remove_product()
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
            print("Exiting Inventory Management System. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-4.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
