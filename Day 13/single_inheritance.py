# ==============================================================================
# Program    : Single Inheritance Demonstration
# Objective  : Model parent-child class relationships using single inheritance.
# Concept    : Single Inheritance (class Child(Parent))
# Why Used   : Allows child class (Dog) to reuse attributes and methods of parent class (Animal).
# ==============================================================================

# What is used : Parent class definition 'class Animal:'
# Why it is used: Defines base attributes and behaviors common to all animals
# How it works : Allocates Animal class object in Python's module scope
class Animal:
    """Parent base class representing an animal."""

    # What is used : Constructor method '__init__(self, name)'
    # Why it is used: Initializes base attribute 'name' for animal instances
    # How it works : Binds name parameter to instance attribute self.name
    def __init__(self, name):
        # What is used : Instance attribute assignment
        # Why it is used: Stores animal name in instance memory
        self.name = name

    # What is used : Base method 'speak(self)'
    # Why it is used: Provides default generic animal sound behavior
    # How it works : Prints formatted string accessing self.name
    def speak(self):
        print(f"{self.name} makes a generic animal sound.")

# What is used : Child class definition 'class Dog(Animal):'
# Why it is used: Inherits all attributes and methods from Animal parent class
# How it works : Specifies parent class in parentheses after child class name
class Dog(Animal):
    """Child subclass inheriting from Animal."""

    # What is used : Subclass method 'bark(self)'
    # Why it is used: Adds dog-specific barking behavior to child class
    # How it works : Accesses inherited self.name attribute from parent class
    def bark(self):
        print(f"{self.name} barks: Woof! Woof!")

def main():
    print("=== Single Inheritance Demonstration ===")

    # What is used : Object instantiation of child class 'Dog("Rover")'
    # Why it is used: Creates concrete Dog instance in memory
    # How it works : Calls inherited Animal __init__ constructor automatically
    my_dog = Dog("Rover")

    # What is used : Invoking inherited parent method 'my_dog.speak()'
    # How it works : Looks up speak method in Dog class, finds it in parent Animal, executes Animal.speak(my_dog)
    my_dog.speak()

    # What is used : Invoking child-specific method 'my_dog.bark()'
    # How it works : Executes Dog.bark(my_dog) directly from Dog subclass
    my_dog.bark()

if __name__ == "__main__":
    main()
