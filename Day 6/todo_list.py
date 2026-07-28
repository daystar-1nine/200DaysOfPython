# ==============================================================================
# Program    : Bonus Challenge: To-Do List Application
# Objective  : Practice and master bonus challenge: to-do list application logic.
# Concept    : Add task, Remove task, Mark task completed, View tasks, Exit
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Executes continuously as long as the specified boolean condition remains True.
# ==============================================================================


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():
    # Each task is stored as a dictionary: {"title": task_name, "completed": True/False}
    tasks = [
        {"title": "Learn Python Lists", "completed": True},
        {"title": "Practice 10 Coding Problems", "completed": False},
        {"title": "Build To-Do List App", "completed": False}
    ]


# What is used : while loop condition
# Why it is used: Continuously executes code block as long as condition evaluates to True
    while True:
        print("\n--- To-Do List App ---")
        print("1. View Tasks")
        print("2. Add Task")
        print("3. Mark Task as Completed")
        print("4. Remove Task")
        print("5. Exit")

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
        choice = input("Select an option (1-5): ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
        if choice == "1":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not tasks:
                print("No tasks available.")
            else:
                print("\nYour Tasks:")

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
                for idx, task in enumerate(tasks, start=1):
                    status = "[X] Completed" if task["completed"] else "[ ] Pending"
                    print(f"{idx}. {task['title']} - {status}")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "2":
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
            title = input("Enter new task description: ").strip()
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if title:
                tasks.append({"title": title, "completed": False})
                print(f"Task '{title}' added!")
            else:
                print("Task title cannot be empty.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "3":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not tasks:
                print("No tasks to complete.")
                continue
            try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
                task_num = int(input("Enter task number to mark completed: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
                if 1 <= task_num <= len(tasks):
                    tasks[task_num - 1]["completed"] = True
                    print(f"Task '{tasks[task_num - 1]['title']}' marked as completed!")
                else:
                    print("Invalid task number.")
            except ValueError:
                print("Please enter a valid number.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "4":
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
            if not tasks:
                print("No tasks to remove.")
                continue
            try:
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
                task_num = int(input("Enter task number to remove: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
                if 1 <= task_num <= len(tasks):
                    removed = tasks.pop(task_num - 1)
                    print(f"Task '{removed['title']}' removed!")
                else:
                    print("Invalid task number.")
            except ValueError:
                print("Please enter a valid number.")

# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
        elif choice == "5":
            print("Goodbye! Stay productive!")
            break
        else:
            print("Invalid option. Please enter 1-5.")

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
