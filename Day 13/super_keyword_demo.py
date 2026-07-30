# ==============================================================================
# Program    : Super Keyword Demonstration
# Objective  : Delegate parent constructor and method calls using super().
# Concept    : Parent Class Delegation (super().__init__)
# Why Used   : super() calls parent methods without hardcoding explicit parent class names.
# ==============================================================================

# What is used : Base class definition 'class Animal:'
class Animal:
    """Parent base class."""

    def __init__(self, name, species):
        self.name = name
        self.species = species

    def display_basic_info(self):
        print(f"Name: {self.name} | Species: {self.species}")

# What is used : Subclass 'class Dog(Animal):' utilizing super()
class Dog(Animal):
    """Child subclass extending Animal."""

    def __init__(self, name, breed):
        # What is used : super().__init__(name, species="Canine")
        # Why it is used: Delegates name and species initialization to parent Animal __init__
        # How it works : Locates parent class constructor via MRO and passes self context automatically
        super().__init__(name, species="Canine")
        self.breed = breed

    def display_full_info(self):
        # What is used : super().display_basic_info()
        # Why it is used: Executes parent method before adding subclass-specific details
        super().display_basic_info()
        print(f"Breed: {self.breed}")

def main():
    print("=== super() Keyword Demonstration ===")
    dog = Dog("Buddy", "Golden Retriever")
    dog.display_full_info()

if __name__ == "__main__":
    main()
