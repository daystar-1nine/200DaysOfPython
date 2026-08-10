# ==============================================================================
# Program    : Product Dataclass with Custom Methods (Task 5)
# Objective  : Add total_price() method to Product dataclass calculating total inventory value.
# Concept    : Dataclass Methods & Computations
# Why Used   : Encapsulates data attributes and inventory calculation logic within dataclass.
# ==============================================================================

from dataclasses import dataclass

@dataclass
class Product:
    name: str
    price: float
    quantity: int

    # What is used : Instance method inside dataclass
    # Why it is used: Calculates total inventory value for the product item
    def total_price(self) -> float:
        return self.price * self.quantity

def main() -> None:
    print("=== TASK 5: PRODUCT DATACLASS WITH TOTAL_PRICE METHOD ===")
    
    laptop = Product(name="Dell XPS Laptop", price=85000.0, quantity=4)
    monitor = Product(name="LG 4K Monitor", price=28000.0, quantity=2)

    print(f"Item: {laptop.name:<20} | Price: Rs.{laptop.price:<8} | Qty: {laptop.quantity} | Total: Rs.{laptop.total_price():,.2f}")
    print(f"Item: {monitor.name:<20} | Price: Rs.{monitor.price:<8} | Qty: {monitor.quantity} | Total: Rs.{monitor.total_price():,.2f}")

if __name__ == "__main__":
    main()
