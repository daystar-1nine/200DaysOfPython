# ==============================================================================
# Program    : Expense Tracker with Integrated File & Exception Logging (Bonus Challenge)
# Objective  : Integrated expense tracker combining File I/O, Exception Handling, and Logging to 'expenses.log'.
# Concept    : Integrated File Handling, Exception Handling & Application Logging
# Why Used   : Tracks expense operations (INFO, WARNING, ERROR) alongside persistent JSON database storage.
# ==============================================================================

import json
import logging
import os

log_file = os.path.join(os.path.dirname(__file__), "expenses.log")
db_file = os.path.join(os.path.dirname(__file__), "expenses_data.json")

# Configure Expense Tracker Logger
logger = logging.getLogger("ExpenseTracker")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class ExpenseTracker:
    def __init__(self):
        self.expenses = self.load_expenses()

    def load_expenses(self):
        if not os.path.exists(db_file):
            return []
        try:
            with open(db_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("File operation failed during load: %s", e, exc_info=True)
            print("Error loading database file. Initializing empty database.")
            return []

    def save_expenses(self):
        try:
            with open(db_file, "w", encoding="utf-8") as f:
                json.dump(self.expenses, f, indent=4)
        except OSError as e:
            logger.error("File operation failed during save: %s", e, exc_info=True)
            print("Error: Could not save expenses to disk.")

    def add_expense(self, category, amount):
        if amount <= 0:
            logger.warning("Invalid amount attempted for category '%s': Rs.%.2f", category, amount)
            print("Warning: Expense amount must be positive.")
            return
        entry = {"category": category, "amount": amount}
        self.expenses.append(entry)
        self.save_expenses()
        logger.info("Expense added | %s | Rs.%.2f", category, amount)
        print(f"Added Expense: {category} -> Rs.{amount:.2f}")

    def delete_expense(self, category):
        initial_len = len(self.expenses)
        self.expenses = [e for e in self.expenses if e["category"].lower() != category.lower()]
        if len(self.expenses) < initial_len:
            self.save_expenses()
            logger.info("Expense deleted | %s", category)
            print(f"Deleted Expense category: {category}")
        else:
            logger.warning("Delete failed: Category '%s' not found", category)
            print(f"Warning: Category '{category}' not found.")

def main():
    print("==========================================================")
    print("        EXPENSE TRACKER WITH INTEGRATED LOGGING           ")
    print("==========================================================")

    tracker = ExpenseTracker()
    tracker.add_expense("Food", 250.0)
    tracker.add_expense("Travel", 120.0)
    tracker.add_expense("Books", 450.0)
    tracker.add_expense("Gadgets", -50.0)  # Invalid amount
    tracker.delete_expense("Books")
    tracker.delete_expense("NonExistent")   # Warning

    print(f"\nAll operations logged to '{log_file}' & database saved to '{db_file}'.")

if __name__ == "__main__":
    main()
