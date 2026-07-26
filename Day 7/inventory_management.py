# Bonus Challenge: Inventory Management System
# Features: Product, Quantity, Price, Calculate total inventory value

inventory = {
    "Laptop": {"quantity": 5, "price": 65000},
    "Mouse": {"quantity": 20, "price": 800},
    "Keyboard": {"quantity": 15, "price": 1200}
}

def display_inventory():
    print("\n=======================================================")
    print("              INVENTORY MANAGEMENT SYSTEM")
    print("=======================================================")
    print(f"{'Product':<15} {'Quantity':<10} {'Price (₹)':<12} {'Total Value (₹)':<15}")
    print("-------------------------------------------------------")
    
    grand_total = 0
    for product, details in inventory.items():
        qty = details["quantity"]
        price = details["price"]
        total = qty * price
        grand_total += total
        print(f"{product:<15} {qty:<10} {price:<12} {total:<15}")
    
    print("-------------------------------------------------------")
    print(f"GRAND TOTAL INVENTORY VALUE: ₹{grand_total:,}")
    print("=======================================================")

def add_or_update_product():
    product = input("Enter Product Name: ").strip()
    if not product:
        print("Product name cannot be empty.")
        return
    try:
        qty = int(input("Enter Quantity: "))
        price = float(input("Enter Price per unit (₹): "))
        
        inventory[product] = {"quantity": qty, "price": price}
        print(f"Product '{product}' added/updated successfully!")
    except ValueError:
        print("Invalid input! Quantity and price must be numbers.")

def remove_product():
    product = input("Enter Product Name to remove: ").strip()
    if product in inventory:
        del inventory[product]
        print(f"Product '{product}' removed from inventory.")
    else:
        print("Product not found.")

def main():
    while True:
        print("\n--- Inventory Menu ---")
        print("1. Display Inventory & Total Value")
        print("2. Add / Update Product")
        print("3. Remove Product")
        print("4. Exit")

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            display_inventory()
        elif choice == "2":
            add_or_update_product()
        elif choice == "3":
            remove_product()
        elif choice == "4":
            print("Exiting Inventory Management System. Goodbye!")
            break
        else:
            print("Invalid choice! Enter 1-4.")

if __name__ == "__main__":
    main()
