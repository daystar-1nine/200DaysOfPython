# ==============================================================================
# Program    : Challenge Project: Phone Book Application
# Objective  : Practice and master challenge project: phone book application logic.
# Concept    : Core Concepts
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================

# Menu: 1 Add Contact, 2 Search Contact, 3 Delete Contact, 4 Display All, 5 Exit

contacts = {
    "Amit": "9876543210",
    "Rahul": "9123456780"
}


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():

# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n--- Phone Book Application ---")
        print("1. Add Contact")
        print("2. Search Contact")
        print("3. Delete Contact")
        print("4. Display All Contacts")
        print("5. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select an option (1-5): ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            name = input("Enter Contact Name: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not name:
                print("Name cannot be empty.")
                continue
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            phone = input("Enter Phone Number: ").strip()
            contacts[name] = phone
            print(f"Contact '{name}' added/updated successfully!")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            name = input("Enter Contact Name to search: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if name in contacts:
                print(f"Contact {name}: {contacts[name]}")
            else:
                print(f"Contact '{name}' not found.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            name = input("Enter Contact Name to delete: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if name in contacts:
                deleted_num = contacts.pop(name)
                print(f"Deleted contact '{name}' ({deleted_num}).")
            else:
                print(f"Contact '{name}' not found.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not contacts:
                print("Phone book is empty.")
            else:
                print("\n------ CONTACT LIST ------")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
                for name, phone in contacts.items():
                    print(f"Contact {name:<15} : {phone}")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "5":
            print("Exiting Phone Book. Goodbye!")
            break
        else:
            print("Invalid selection! Please enter 1-5.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
