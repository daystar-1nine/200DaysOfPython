# Bonus Challenge: To-Do List Application
# Features: Add task, Remove task, Mark task completed, View tasks, Exit

def main():
    # Each task is stored as a dictionary: {"title": task_name, "completed": True/False}
    tasks = [
        {"title": "Learn Python Lists", "completed": True},
        {"title": "Practice 10 Coding Problems", "completed": False},
        {"title": "Build To-Do List App", "completed": False}
    ]

    while True:
        print("\n--- 📝 To-Do List App ---")
        print("1. View Tasks")
        print("2. Add Task")
        print("3. Mark Task as Completed")
        print("4. Remove Task")
        print("5. Exit")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            if not tasks:
                print("No tasks available.")
            else:
                print("\nYour Tasks:")
                for idx, task in enumerate(tasks, start=1):
                    status = "[✓] Completed" if task["completed"] else "[ ] Pending"
                    print(f"{idx}. {task['title']} - {status}")

        elif choice == "2":
            title = input("Enter new task description: ").strip()
            if title:
                tasks.append({"title": title, "completed": False})
                print(f"Task '{title}' added!")
            else:
                print("Task title cannot be empty.")

        elif choice == "3":
            if not tasks:
                print("No tasks to complete.")
                continue
            try:
                task_num = int(input("Enter task number to mark completed: "))
                if 1 <= task_num <= len(tasks):
                    tasks[task_num - 1]["completed"] = True
                    print(f"Task '{tasks[task_num - 1]['title']}' marked as completed!")
                else:
                    print("Invalid task number.")
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "4":
            if not tasks:
                print("No tasks to remove.")
                continue
            try:
                task_num = int(input("Enter task number to remove: "))
                if 1 <= task_num <= len(tasks):
                    removed = tasks.pop(task_num - 1)
                    print(f"Task '{removed['title']}' removed!")
                else:
                    print("Invalid task number.")
            except ValueError:
                print("Please enter a valid number.")

        elif choice == "5":
            print("Goodbye! Stay productive! 🚀")
            break
        else:
            print("Invalid option. Please enter 1-5.")

if __name__ == "__main__":
    main()
