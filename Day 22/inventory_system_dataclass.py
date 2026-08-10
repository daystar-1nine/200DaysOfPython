# ==============================================================================
# Program    : Inventory Management System (Challenge Project)
# Objective  : Manage product inventory via dataclass with stock modification methods.
# Concept    : Dataclass State Mutation & Error Validation
# Why Used   : Manages stock additions, removals, and calculates total inventory value.
# ==============================================================================

from dataclasses import dataclass

@dataclass
class Product:
    id: int
    name: str
    price: float
    quantity: int

    def total_value(self) -> float:
        return self.price * self.quantity

    def add_stock(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("Stock addition amount must be strictly positive.")
        self.quantity += amount
        print(f"[STOCK ADDED] Added {amount} units to {self.name}. New Quantity: {self.quantity}")

    def remove_stock(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError("Stock removal amount must be strictly positive.")
        if amount > self.quantity:
            raise ValueError(f"Insufficient stock for {self.name}! Available: {self.quantity}, Requested: {amount}")
        self.quantity -= amount
        print(f"[STOCK REMOVED] Removed {amount} units from {self.name}. New Quantity: {self.quantity}")

def main() -> None:
    print("==========================================================")
    print("              INVENTORY MANAGEMENT SYSTEM                 ")
    print("==========================================================")

    p1 = Product(id=501, name="Laptop", price=65000.0, quantity=3)
    
    print(f"\nProduct    : {p1.name}")
    print(f"Price      : Rs.{p1.price:,.2f}")
    print(f"Quantity   : {p1.quantity}")
    print(f"Total Value: Rs.{p1.total_value():,.2f}\n")

    p1.add_stock(5)
    print(f"Updated Total Value: Rs.{p1.total_value():,.2f}\n")

    try:
        p1.remove_stock(2)
        p1.remove_stock(10)  # Should trigger ValueError
    except ValueError as e:
        print(f"[VALIDATION ERROR] {e}")

if __name__ == "__main__":
    main()
