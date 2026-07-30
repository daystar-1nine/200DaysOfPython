# ==============================================================================
# Program    : Name Memory Closure
# Objective  : Demonstrate closure function that remembers a user's name across calls.
# Concept    : Closures (Free Variables & Enclosing Scope Persistence)
# Why Used   : Inner function encapsulates name variable from outer function scope permanently.
# ==============================================================================

# What is used : Outer function 'def name_rememberer(name)'
# Why it is used: Encloses parameter 'name' in its local scope
def name_rememberer(name):
    # What is used : Inner function 'def greeter()'
    # How it works : References 'name' from enclosing scope (Free Variable)
    def greeter(custom_message="Hello"):
        return f"{custom_message}, {name}!"
    
    # What is used : Returning inner function object without calling it ()
    return greeter

def main():
    print("=== Name Closure Demonstration ===")
    
    # What is used : Instantiating closure functions
    suraj_greeter = name_rememberer("Suraj")
    rahul_greeter = name_rememberer("Rahul")

    # Executing closures after outer function has returned
    print(suraj_greeter("Good morning"))
    print(rahul_greeter("Welcome back"))

    # What is used : Inspecting closure cell object
    print(f"\nBound Name in Closure Cell: {suraj_greeter.__closure__[0].cell_contents}")

if __name__ == "__main__":
    main()
