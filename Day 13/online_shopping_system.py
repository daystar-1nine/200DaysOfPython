# ==============================================================================
# Program    : Online Shopping System (Advanced OOP Integration)
# Objective  : Model e-commerce catalog with inheritance, price encapsulation, and polymorphic checkout.
# Concept    : Encapsulation (Private __price), Method Overriding & Polymorphic Checkout
# Why Used   : Encapsulates product prices, overrides discount math, and processes shopping cart checkout.
# ==============================================================================

# What is used : Base Class 'class Product:'
class Product:
    """Base Product class encapsulating price and discount logic."""

    def __init__(self, product_id, name, price):
        self.product_id = product_id
        self.name = name
        # Private attribute for price encapsulation
        self.__price = max(0.0, price)

    # Getter for private price
    def get_price(self):
        return self.__price

    # Setter with validation
    def set_price(self, new_price):
        if new_price >= 0:
            self.__price = new_price
        else:
            print("Price Error: Price cannot be negative!")

    # Base discount method (0% default discount)
    def calculate_discounted_price(self):
        return self.get_price()

    def get_details(self):
        return f"[{self.product_id}] {self.name:<20} | Price: Rs.{self.get_price():,.2f} | Final: Rs.{self.calculate_discounted_price():,.2f}"

# What is used : Subclass Electronics with 10% tech discount
class Electronics(Product):
    def __init__(self, product_id, name, price, warranty_years=1):
        super().__init__(product_id, name, price)
        self.warranty_years = warranty_years

    # Method Overriding for 10% discount
    def calculate_discounted_price(self):
        return self.get_price() * 0.90

# What is used : Subclass Clothing with 20% fashion discount
class Clothing(Product):
    def __init__(self, product_id, name, price, size):
        super().__init__(product_id, name, price)
        self.size = size

    # Method Overriding for 20% discount
    def calculate_discounted_price(self):
        return self.get_price() * 0.80

# What is used : Subclass Grocery with 5% fresh grocery discount
class Grocery(Product):
    def __init__(self, product_id, name, price, expiry_days):
        super().__init__(product_id, name, price)
        self.expiry_days = expiry_days

    # Method Overriding for 5% discount
    def calculate_discounted_price(self):
        return self.get_price() * 0.95

# What is used : Cart container class using Polymorphism
class ShoppingCart:
    """Shopping Cart container processing polymorphic checkout."""

    def __init__(self):
        self.cart = []

    def add_product(self, product_obj):
        self.cart.append(product_obj)
        print(f"Added '{product_obj.name}' to cart.")

    # Polymorphic Checkout Method
    def checkout(self):
        if not self.cart:
            print("Cart is empty!")
            return

        print("\n==========================================================================")
        print("                        ONLINE SHOPPING CHECKOUT                           ")
        print("==========================================================================")
        total_original = 0.0
        total_final = 0.0

        for item in self.cart:
            print(item.get_details())
            total_original += item.get_price()
            total_final += item.calculate_discounted_price()

        savings = total_original - total_final
        print("--------------------------------------------------------------------------")
        print(f"Total Original Price : Rs.{total_original:,.2f}")
        print(f"Total Discount Savings: Rs.{savings:,.2f}")
        print(f"GRAND TOTAL PAYABLE  : Rs.{total_final:,.2f}")
        print("==========================================================================\n")

def main():
    cart = ShoppingCart()

    p1 = Electronics("P101", "Laptop", 65000, warranty_years=2)
    p2 = Clothing("P201", "Denim Jacket", 3500, size="L")
    p3 = Grocery("P301", "Almond Packet", 800, expiry_days=180)

    cart.add_product(p1)
    cart.add_product(p2)
    cart.add_product(p3)

    cart.checkout()

if __name__ == "__main__":
    main()
