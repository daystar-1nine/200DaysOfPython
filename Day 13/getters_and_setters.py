# ==============================================================================
# Program    : Getters and Setters Demonstration
# Objective  : Provide controlled accessor and mutator methods with data validation.
# Concept    : Getters (Accessors) and Setters (Mutators)
# Why Used   : Setters enforce data boundary rules (e.g. marks between 0 and 100) before updating private attributes.
# ==============================================================================

# What is used : Encapsulated class definition 'class Student:'
class Student:
    """Class using Getters and Setters for private marks attribute."""

    def __init__(self, name, initial_marks=0):
        self.name = name
        # Private attribute
        self.__marks = 0
        # Utilize setter for initial validation
        self.set_marks(initial_marks)

    # What is used : Getter method 'get_marks(self)'
    # Why it is used: Returns private attribute self.__marks safely
    def get_marks(self):
        return self.__marks

    # What is used : Setter method 'set_marks(self, marks)'
    # Why it is used: Validates that marks fall within legal academic range [0, 100]
    # How it works : Updates self.__marks only if validation condition is satisfied
    def set_marks(self, marks):
        if 0 <= marks <= 100:
            self.__marks = marks
            print(f"Marks for {self.name} updated to: {self.__marks}")
        else:
            print(f"Validation Error: Marks '{marks}' invalid! Must be between 0 and 100.")

def main():
    print("=== Getters and Setters Demonstration ===")
    student = Student("Suraj", 85)

    print(f"Current Marks via Getter: {student.get_marks()}")

    print("\n[Action] Updating marks to valid score (95)...")
    student.set_marks(95)

    print("\n[Action] Attempting to set invalid score (150)...")
    student.set_marks(150)

    print(f"\nFinal Marks via Getter: {student.get_marks()}")

if __name__ == "__main__":
    main()
