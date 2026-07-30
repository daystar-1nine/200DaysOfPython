# ==============================================================================
# Program    : Method Overriding Demonstration
# Objective  : Override parent class method implementation in child subclasses.
# Concept    : Method Overriding (Polymorphic Behavior Replacement)
# Why Used   : Allows child classes to provide specialized implementations of inherited methods.
# ==============================================================================

# What is used : Parent class definition 'class Animal:'
class Animal:
    """Parent base class."""

    def speak(self):
        print("Animal speaks a generic sound.")

# What is used : Subclass Dog overriding speak()
class Dog(Animal):
    """Child subclass Dog."""

    # What is used : Method overriding
    # Why it is used: Replaces parent implementation of speak() with dog-specific barking behavior
    # How it works : When called on a Dog instance, Python finds and executes Dog.speak() instead of Animal.speak()
    def speak(self):
        print("Dog barks: Woof! Woof!")

# What is used : Subclass Cat overriding speak()
class Cat(Animal):
    """Child subclass Cat."""

    def speak(self):
        print("Cat meows: Meow! Meow!")

def main():
    print("=== Method Overriding Demonstration ===")
    generic_animal = Animal()
    dog = Dog()
    cat = Cat()

    print("Generic Animal:")
    generic_animal.speak()

    print("\nOverridden Subclass Methods:")
    dog.speak()
    cat.speak()

if __name__ == "__main__":
    main()
