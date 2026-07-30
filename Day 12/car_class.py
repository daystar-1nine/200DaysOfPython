# ==============================================================================
# Program    : Car Class Implementation
# Objective  : Model automobiles using OOP attributes and behavior methods.
# Concept    : Classes, Instances, Instance Attributes & Methods
# Why Used   : Combines physical vehicle properties (brand, model, price) and actions (start_engine, display).
# ==============================================================================

# What is used : Class definition 'class Car:'
# Why it is used: Acts as blueprint for manufacturing individual car objects
# How it works : Defines shared structure for all car instances
class Car:
    """Class representing an automobile."""

    # What is used : Constructor __init__ with required parameters (brand, model, price)
    # Why it is used: Initializes car properties upon instantiation
    # How it works : Sets self.brand, self.model, self.price to arguments passed
    def __init__(self, brand, model, price):
        # What is used : Instance attribute binding via self
        self.brand = brand
        self.model = model
        self.price = price

    # What is used : Instance method 'start_engine(self)'
    # Why it is used: Models car behavior/action
    # How it works : Reads instance brand and model to print ignition message
    def start_engine(self):
        print(f"Vroom! The engine of {self.brand} {self.model} is now RUNNING!")

    # What is used : Instance method 'display_details(self)'
    # Why it is used: Displays automobile specifications with currency formatting
    def display_details(self):
        print("\n--- CAR SPECIFICATIONS ---")
        print(f"Brand : {self.brand}")
        print(f"Model : {self.model}")
        print(f"Price : Rs.{self.price:,.2f}")

def main():
    # What is used : Object Instantiations
    # Why it is used: Creates two distinct Car objects with unique state data
    car1 = Car("Tesla", "Model 3", 4500000)
    car2 = Car("BMW", "M5 CS", 12000000)

    # What is used : Method Invocations
    # How it works : Executes methods using specific instance state
    car1.display_details()
    car1.start_engine()

    car2.display_details()
    car2.start_engine()

if __name__ == "__main__":
    main()
