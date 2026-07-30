# ==============================================================================
# Program    : Multiple Inheritance Demonstration
# Objective  : Model a child class inheriting behaviors from multiple parent classes.
# Concept    : Multiple Inheritance (class Child(Father, Mother)) & MRO
# Why Used   : Enables a subclass to aggregate methods from two independent base classes.
# ==============================================================================

# What is used : First Parent class definition 'class Father:'
# Why it is used: Encapsulates father-specific attributes and behaviors
class Father:
    """First parent class."""

    def __init__(self, father_name="Suresh"):
        # What is used : Instance attribute assignment
        self.father_name = father_name

    # What is used : Father behavior method 'gardening(self)'
    def gardening(self):
        print(f"Father ({self.father_name}) loves gardening.")

# What is used : Second Parent class definition 'class Mother:'
# Why it is used: Encapsulates mother-specific attributes and behaviors
class Mother:
    """Second parent class."""

    def __init__(self, mother_name="Sunita"):
        # What is used : Instance attribute assignment
        self.mother_name = mother_name

    # What is used : Mother behavior method 'cooking(self)'
    def cooking(self):
        print(f"Mother ({self.mother_name}) is a master chef in cooking.")

# What is used : Child class inheriting from BOTH parents 'class Child(Father, Mother):'
# Why it is used: Inherits methods and properties from both Father and Mother classes
# How it works : Python searches methods according to Method Resolution Order (MRO)
class Child(Father, Mother):
    """Child class inheriting from Father and Mother."""

    def __init__(self, child_name, father_name, mother_name):
        # What is used : Direct parent constructor initializations
        # Why it is used: Ensures attributes of both parent classes are initialized properly
        Father.__init__(self, father_name)
        Mother.__init__(self, mother_name)
        self.child_name = child_name

    # What is used : Child method 'hobbies(self)'
    # Why it is used: Demonstrates calling inherited methods from both parents
    def hobbies(self):
        print(f"\n{self.child_name}'s Inherited Skills:")
        self.gardening()
        self.cooking()

def main():
    print("=== Multiple Inheritance Demonstration ===")
    child = Child("Suraj", "Suresh", "Sunita")
    child.hobbies()

    # What is used : Class Method Resolution Order 'Child.mro()'
    # Why it is used: Inspects the resolution order Python follows during method lookup
    print(f"\nMethod Resolution Order (MRO):\n{[cls.__name__ for cls in Child.mro()]}")

if __name__ == "__main__":
    main()
