# ==============================================================================
# Program    : Bank Account OOP System
# Objective  : Model bank accounts with deposit, withdrawal, and validation methods.
# Concept    : OOP State Management & Boundary Validation
# Why Used   : BankAccount class handles balance updates and prevents negative deposits/insufficient funds.
# ==============================================================================

# What is used : Class definition 'class BankAccount:'
class BankAccount:
    """Class representing a customer bank account."""

    def __init__(self, name, account_number, initial_balance=0.0):
        # What is used : Instance attributes initialization
        self.name = name
        self.account_number = account_number
        self.balance = initial_balance

    # What is used : Instance method 'deposit(self, amount)'
    # Why it is used: Validates deposit amount > 0 and adds to self.balance
    def deposit(self, amount):
        if amount <= 0:
            print("Deposit Error: Amount must be greater than zero!")
            return False
        self.balance += amount
        print(f"[Success] Rs.{amount:,.2f} deposited. New Balance: Rs.{self.balance:,.2f}")
        return True

    # What is used : Instance method 'withdraw(self, amount)'
    # Why it is used: Validates withdrawal amount <= self.balance and subtracts from balance
    def withdraw(self, amount):
        if amount <= 0:
            print("Withdrawal Error: Amount must be greater than zero!")
            return False
        if amount > self.balance:
            print(f"Withdrawal Error: Insufficient funds! Available: Rs.{self.balance:,.2f}, Requested: Rs.{amount:,.2f}")
            return False
        self.balance -= amount
        print(f"[Success] Rs.{amount:,.2f} withdrawn. Remaining Balance: Rs.{self.balance:,.2f}")
        return True

    # What is used : Instance method 'check_balance(self)'
    # Why it is used: Outputs current account holder and balance summary
    def check_balance(self):
        print(f"\n--- Account Summary ({self.account_number}) ---")
        print(f"Account Holder: {self.name}")
        print(f"Current Balance: Rs.{self.balance:,.2f}")

def main():
    account = BankAccount("Suraj Sawant", "ACC-987654", 10000.0)
    account.check_balance()

    print("\n[Action] Depositing Rs.5,000...")
    account.deposit(5000)

    print("\n[Action] Attempting invalid negative deposit Rs.-500...")
    account.deposit(-500)

    print("\n[Action] Withdrawing Rs.3,000...")
    account.withdraw(3000)

    print("\n[Action] Attempting over-withdrawal Rs.20,000...")
    account.withdraw(20000)

    account.check_balance()

if __name__ == "__main__":
    main()
