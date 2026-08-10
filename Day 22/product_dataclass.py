# ==============================================================================
# Program    : Product Dataclass Definition (Task 4)
# Objective  : Define a basic Product dataclass with fields name, price, and quantity.
# Concept    : Python Dataclasses (@dataclass)
# Why Used   : Automatically generates __init__, __repr__, and __eq__ boilerplate methods.
# ==============================================================================

from dataclasses import dataclass

# What is used : @dataclass decorator
# Why it is used: Transforms class into a clean data container with auto-generated constructor
@dataclass
class Product:
    name: str
    price: float
    quantity: int

def main() -> None:
    print("=== TASK 4: PRODUCT DATACLASS DEMO ===")
    
    # Instantiate Product instances
    p1 = Product(name="Gaming Mouse", price=1200.0, quantity=5)
    p2 = Product(name="Mechanical Keyboard", price=3500.0, quantity=3)

    print("Product 1 Instance:", p1)
    print("Product 2 Instance:", p2)
    print(f"Product 1 Name: {p1.name}, Price: Rs.{p1.price}")

if __name__ == "__main__":
    main()
