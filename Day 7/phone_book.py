# Challenge Project: Phone Book Application
# Menu: 1 Add Contact, 2 Search Contact, 3 Delete Contact, 4 Display All, 5 Exit

contacts = {
    "Amit": "9876543210",
    "Rahul": "9123456780"
}

def main():
    while True:
        print("\n--- 📞 Phone Book Application ---")
        print("1. Add Contact")
        print("2. Search Contact")
        print("3. Delete Contact")
        print("4. Display All Contacts")
        print("5. Exit")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            name = input("Enter Contact Name: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            phone = input("Enter Phone Number: ").strip()
            contacts[name] = phone
            print(f"Contact '{name}' added/updated successfully!")

        elif choice == "2":
            name = input("Enter Contact Name to search: ").strip()
            if name in contacts:
                print(f"📞 {name}: {contacts[name]}")
            else:
                print(f"Contact '{name}' not found.")

        elif choice == "3":
            name = input("Enter Contact Name to delete: ").strip()
            if name in contacts:
                deleted_num = contacts.pop(name)
                print(f"Deleted contact '{name}' ({deleted_num}).")
            else:
                print(f"Contact '{name}' not found.")

        elif choice == "4":
            if not contacts:
                print("Phone book is empty.")
            else:
                print("\n------ CONTACT LIST ------")
                for name, phone in contacts.items():
                    print(f"👤 {name:<15} : 📞 {phone}")

        elif choice == "5":
            print("Exiting Phone Book. Goodbye!")
            break
        else:
            print("Invalid selection! Please enter 1-5.")

if __name__ == "__main__":
    main()
