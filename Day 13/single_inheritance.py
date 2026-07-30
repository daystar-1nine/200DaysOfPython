# ==============================================================================
# Program    : Single Inheritance Demonstration
# Objective  : Model parent-child class relationships using single inheritance.
# Concept    : Single Inheritance (class Child(Parent))
# Why Used   : Allows child class (Dog) to reuse attributes and methods of parent class (Animal).
# ==============================================================================

# What is used : Parent class definition 'class Animal:'
# Why it is used: Defines base attributes and behaviors common to all animals
class Animal:
    """Parent base class representing an animal."""

    def __init__(self, name):
        # What is used : Base attribute assignment
        self.name = name

    # What is used : Base method 'speak(self)'
    def speak(self):
        print(f"{self.name} makes a generic animal sound.")

# What is used : Child class definition 'class Dog(Animal):'
# Why it is used: Inherits all attributes and methods from Animal parent class
# How it works : Specifies parent class in parentheses after child class name
class Dog(Animal):
    """Child subclass inheriting from Animal."""

    def bark(self):
        print(f"{self.name} barks: Woof! Woof!")

def main():
    print("=== Single Inheritance Demonstration ===")

    # What is used : Object instantiation of child class 'Dog("Rover")'
    # How it works : Calls inherited Animal __init__ constructor automatically
    my_dog = Dog("Rover")

    # What is used : Invoking inherited parent method 'my_dog.speak()'
    my_dog.speak()

    # What is used : Invoking child-specific method 'my_dog.bark()'
    my_dog.bark()

if __name__ == "__main__":
    main()
