# ==============================================================================
# Program    : Challenge Project: Shopping Cart CLI Application
# Objective  : Practice and master challenge project: shopping cart cli application logic.
# Concept    : Core Concepts
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# Menu: 1. Add Item, 2. Remove Item, 3. View Cart, 4. Search Item, 5. Exit

def main():
    cart = ["Apple", "Milk", "Bread", "Eggs"]  # Pre-populated sample items

    while True:
        print("\n--- Shopping Cart ---")
        print("1. Add Item")
        print("2. Remove Item")
        print("3. View Cart")
        print("4. Search Item")
        print("5. Exit")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            item = input("Enter item name to add: ").strip()
            if item:
                cart.append(item)
                print(f"'{item}' added to cart!")
            else:
                print("Item name cannot be empty.")

        elif choice == "2":
            item = input("Enter item name to remove: ").strip()
            if item in cart:
                cart.remove(item)
                print(f"'{item}' removed from cart!")
            else:
                print(f"'{item}' not found in cart.")

        elif choice == "3":
            if not cart:
                print("Your cart is empty.")
            else:
                print("\nCart Items:")
                for i, item in enumerate(cart, start=1):
                    print(f"{i}. {item}")

        elif choice == "4":
            query = input("Enter item to search: ").strip()
            if query in cart:
                print(f"'{query}' is in your cart (Item #{cart.index(query) + 1}).")
            else:
                print(f"'{query}' is NOT in your cart.")

        elif choice == "5":
            print("Thank you for shopping! Goodbye!")
            break
        else:
            print("Invalid option. Please enter 1-5.")

if __name__ == "__main__":
    main()
