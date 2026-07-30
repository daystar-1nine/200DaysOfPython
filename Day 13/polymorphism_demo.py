# ==============================================================================
# Program    : Polymorphism Demonstration
# Objective  : Process heterogeneous objects through a unified method interface.
# Concept    : Polymorphism ("One Interface, Many Implementations" via Duck Typing)
# Why Used   : Allows a loop or function to invoke sound() across Dog, Cat, and Duck instances seamlessly.
# ==============================================================================

# What is used : Independent class definitions with common method 'sound()'
class Dog:
    def sound(self):
        return "Bark! Bark!"

class Cat:
    def sound(self):
        return "Meow! Meow!"

class Duck:
    def sound(self):
        return "Quack! Quack!"

# What is used : Polymorphic function 'make_animal_sound(animal_object)'
# Why it is used: Accepts any object implementing sound() method interface (Duck Typing)
# How it works : Invokes animal_object.sound() dynamically at runtime
def make_animal_sound(animal_object):
    print(f"{type(animal_object).__name__:<6} makes sound -> {animal_object.sound()}")

def main():
    print("=== Polymorphism Demonstration ===")

    # What is used : Heterogeneous list storing instances of different classes
    animals = [Dog(), Cat(), Duck()]

    print("Iterating through polymorphic objects collection:")

    # What is used : for loop executing uniform interface method sound()
    for animal in animals:
        make_animal_sound(animal)

if __name__ == "__main__":
    main()
