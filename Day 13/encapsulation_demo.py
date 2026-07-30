# ==============================================================================
# Program    : Encapsulation Demonstration
# Objective  : Protect internal data using private attributes (`__balance`).
# Concept    : Encapsulation (Private Attributes & Accessor Methods)
# Why Used   : Restricts direct attribute mutation to preserve state integrity and prevent invalid values.
# ==============================================================================

# What is used : Encapsulated class definition 'class BankAccount:'
class BankAccount:
    """Class encapsulating private financial attributes."""

    def __init__(self, account_holder, initial_balance=1000.0):
        self.account_holder = account_holder

        # What is used : Private attribute 'self.__balance' (double underscore prefix)
        # Why it is used: Restricts direct external access and modification (name mangling)
        # How it works : Python renames variable to _BankAccount__balance internally
        self.__balance = initial_balance

    # What is used : Getter accessor method 'get_balance(self)'
    # Why it is used: Provides controlled read-only access to private attribute
    def get_balance(self):
        return self.__balance

    # What is used : Controlled deposit mutator method
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            print(f"Deposited Rs.{amount:,.2f}. New Balance: Rs.{self.__balance:,.2f}")
        else:
            print("Deposit Error: Amount must be positive!")

def main():
    print("=== Encapsulation Demonstration ===")
    account = BankAccount("Suraj Sawant", 5000.0)

    # What is used : Accessing public getter method
    print(f"Account Balance via Getter: Rs.{account.get_balance():,.2f}")

    account.deposit(2500)

    # What is used : Attempting direct private attribute access (Demonstrating protection)
    try:
        print(account.__balance)
    except AttributeError:
        print("\n[Protection Active] Direct access to 'account.__balance' raised AttributeError (Name Mangled)!")

if __name__ == "__main__":
    main()
