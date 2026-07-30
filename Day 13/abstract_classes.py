# ==============================================================================
# Program    : Abstract Classes Demonstration
# Objective  : Enforce common subclass interface contracts using ABC and @abstractmethod.
# Concept    : Abstraction (Abstract Base Classes - ABC module)
# Why Used   : Abstract classes cannot be instantiated directly; they mandate that all subclasses implement abstract methods.
# ==============================================================================

# What is used : Imports from built-in module 'from abc import ABC, abstractmethod'
# Why it is used: Provides ABC base class and @abstractmethod decorator
from abc import ABC, abstractmethod
import math

# What is used : Abstract Base Class 'class Shape(ABC):'
# Why it is used: Defines architectural blueprint contract for all geometric shapes
class Shape(ABC):
    """Abstract Base Class for Geometric Shapes."""

    # What is used : Decorator '@abstractmethod'
    # Why it is used: Forces concrete subclasses to provide an area() implementation
    # How it works : Prevents subclass instantiation if area() method is not overridden
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

# What is used : Concrete subclass 'class Circle(Shape):'
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    # Implementing mandatory abstract area() method
    def area(self):
        return math.pi * (self.radius ** 2)

    # Implementing mandatory abstract perimeter() method
    def perimeter(self):
        return 2 * math.pi * self.radius

# What is used : Concrete subclass 'class Rectangle(Shape):'
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

def main():
    print("=== Abstract Classes Demonstration ===")

    # Instantiating concrete subclasses
    circle = Circle(7)
    rectangle = Rectangle(10, 5)

    shapes = [circle, rectangle]

    for s in shapes:
        print(f"{type(s).__name__:<10} | Area: {s.area():<10.2f} | Perimeter: {s.perimeter():.2f}")

    # Demonstrating instantiation prevention of Abstract Class
    try:
        abstract_shape = Shape()
    except TypeError as e:
        print(f"\n[Abstraction Enforced] {e}")

if __name__ == "__main__":
    main()
