# ==============================================================================
# Program    : Challenge Project: Shopping Cart CLI Application
# Objective  : Practice and master challenge project: shopping cart cli application logic.
# Concept    : Core Concepts
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# Menu: 1. Add Item, 2. Remove Item, 3. View Cart, 4. Search Item, 5. Exit


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():
    cart = ["Apple", "Milk", "Bread", "Eggs"]  # Pre-populated sample items


# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n--- Shopping Cart ---")
        print("1. Add Item")
        print("2. Remove Item")
        print("3. View Cart")
        print("4. Search Item")
        print("5. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select an option (1-5): ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            item = input("Enter item name to add: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if item:
                cart.append(item)
                print(f"'{item}' added to cart!")
            else:
                print("Item name cannot be empty.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            item = input("Enter item name to remove: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if item in cart:
                cart.remove(item)
                print(f"'{item}' removed from cart!")
            else:
                print(f"'{item}' not found in cart.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not cart:
                print("Your cart is empty.")
            else:
                print("\nCart Items:")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
                for i, item in enumerate(cart, start=1):
                    print(f"{i}. {item}")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            query = input("Enter item to search: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if query in cart:
                print(f"'{query}' is in your cart (Item #{cart.index(query) + 1}).")
            else:
                print(f"'{query}' is NOT in your cart.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "5":
            print("Thank you for shopping! Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1-5.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
