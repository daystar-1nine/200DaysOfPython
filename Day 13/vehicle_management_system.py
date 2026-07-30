# ==============================================================================
# Program    : Vehicle Management System
# Objective  : Model vehicle fleets using polymorphism and method overriding.
# Concept    : Inheritance, Method Overriding & Polymorphic Fleet Execution
# Why Used   : Car, Bike, and Truck subclasses override start_engine() and calculate_mileage().
# ==============================================================================

# What is used : Parent Base Class 'class Vehicle:'
class Vehicle:
    """Base Vehicle class."""

    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def start_engine(self):
        print(f"[Vehicle] {self.brand} {self.model} engine started.")

    def stop_engine(self):
        print(f"[Vehicle] {self.brand} {self.model} engine stopped.")

    def calculate_mileage(self):
        return 0.0

# What is used : Subclass Car overriding start_engine and calculate_mileage
class Car(Vehicle):
    def __init__(self, brand, model, fuel_liters, km_driven):
        super().__init__(brand, model)
        self.fuel_liters = fuel_liters
        self.km_driven = km_driven

    def start_engine(self):
        print(f"Car Started [Car {self.brand} {self.model}]")

    def calculate_mileage(self):
        return self.km_driven / self.fuel_liters if self.fuel_liters > 0 else 0.0

# What is used : Subclass Bike overriding start_engine and calculate_mileage
class Bike(Vehicle):
    def __init__(self, brand, model, fuel_liters, km_driven):
        super().__init__(brand, model)
        self.fuel_liters = fuel_liters
        self.km_driven = km_driven

    def start_engine(self):
        print(f"Bike Started [Bike {self.brand} {self.model}]")

    def calculate_mileage(self):
        return self.km_driven / self.fuel_liters if self.fuel_liters > 0 else 0.0

# What is used : Subclass Truck overriding start_engine and calculate_mileage
class Truck(Vehicle):
    def __init__(self, brand, model, fuel_liters, km_driven, cargo_weight_tons):
        super().__init__(brand, model)
        self.fuel_liters = fuel_liters
        self.km_driven = km_driven
        self.cargo_weight_tons = cargo_weight_tons

    def start_engine(self):
        print(f"Truck Started [Truck {self.brand} {self.model} ({self.cargo_weight_tons} Tons Cargo)]")

    def calculate_mileage(self):
        return self.km_driven / self.fuel_liters if self.fuel_liters > 0 else 0.0

def main():
    print("=== VEHICLE MANAGEMENT FLEET ===")

    # Polymorphic list of vehicles
    fleet = [
        Car("Tesla", "Model 3", fuel_liters=40, km_driven=600),
        Bike("Yamaha", "R15", fuel_liters=10, km_driven=450),
        Truck("Volvo", "FH16", fuel_liters=150, km_driven=600, cargo_weight_tons=25)
    ]

    print("\n--- Starting All Fleet Engines (Polymorphic Calls) ---")
    for vehicle in fleet:
        # What is used : Polymorphic method calls
        vehicle.start_engine()
        print(f"  -> Calculated Mileage: {vehicle.calculate_mileage():.2f} km/l\n")

if __name__ == "__main__":
    main()
