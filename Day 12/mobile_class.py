# ==============================================================================
# Program    : Mobile Class Implementation
# Objective  : Model smartphone specifications using OOP.
# Concept    : Class Attributes & Instantiation
# Why Used   : Stores brand, model, RAM capacity, and price per mobile device.
# ==============================================================================

# What is used : Class definition 'class Mobile:'
# Why it is used: Blueprint for mobile phone instances
class Mobile:
    """Class representing a mobile phone."""

    # What is used : Constructor __init__(self, brand, model, ram, price)
    # Why it is used: Stores hardware specifications inside instance
    def __init__(self, brand, model, ram, price):
        self.brand = brand
        self.model = model
        self.ram = ram
        self.price = price

    # What is used : Instance method 'display_specs(self)'
    # Why it is used: Formats mobile specifications into a table row
    def display_specs(self):
        print(f"[Mobile] Phone: {self.brand} {self.model:<10} | RAM: {self.ram:<4} | Price: Rs.{self.price:,.2f}")

def main():
    print("=== MOBILE PHONE SPECS ===")
    phone1 = Mobile("Apple", "iPhone 15 Pro", "8GB", 134900)
    phone2 = Mobile("Samsung", "Galaxy S24", "12GB", 119999)

    phone1.display_specs()
    phone2.display_specs()

if __name__ == "__main__":
    main()
