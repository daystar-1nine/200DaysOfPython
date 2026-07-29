# ==============================================================================
# Program    : ATM Simulator System
# Objective  : Handle invalid inputs, negative amounts, and insufficient balance.
# Concept    : Custom Exceptions & Interactive Menu Control Loop
# Why Used   : Ensures robust financial transactions without crashing on user errors.
# ==============================================================================

# What is used : Custom Exception classes for financial constraints
class NegativeAmountError(Exception):
    pass

class InsufficientBalanceError(Exception):
    pass

class InvalidOptionError(Exception):
    pass

class ATMSimulator:
    def __init__(self, initial_balance=5000.0):
        # What is used : Instance attribute balance
        self.balance = initial_balance

    def check_balance(self):
        print(f"\n[Balance] Current Available Balance: Rs.{self.balance:.2f}")

    def deposit(self):
        try:
            amount = float(input("Enter deposit amount: Rs."))
            if amount <= 0:
                # What is used : Explicit exception raising for negative/zero amounts
                raise NegativeAmountError("Error: Deposit amount must be greater than zero!")
            self.balance += amount
            print(f"[Success] Rs.{amount:.2f} deposited successfully! New Balance: Rs.{self.balance:.2f}")
        except ValueError:
            print("Input Error: Please enter a numerical amount!")
        except NegativeAmountError as e:
            print(e)

    def withdraw(self):
        try:
            amount = float(input("Enter withdrawal amount: Rs."))
            if amount <= 0:
                raise NegativeAmountError("Error: Withdrawal amount must be greater than zero!")
            if amount > self.balance:
                raise InsufficientBalanceError(f"Error: Insufficient balance! (Requested Rs.{amount:.2f}, Balance Rs.{self.balance:.2f})")
            self.balance -= amount
            print(f"[Success] Rs.{amount:.2f} withdrawn successfully! Remaining Balance: Rs.{self.balance:.2f}")
        except ValueError:
            print("Input Error: Please enter a numerical amount!")
        except (NegativeAmountError, InsufficientBalanceError) as e:
            print(e)

def main():
    atm = ATMSimulator()
    while True:
        print("\n=== ATM SIMULATOR MENU ===")
        print("1. Check Balance")
        print("2. Deposit Money")
        print("3. Withdraw Money")
        print("4. Exit")

        try:
            choice = input("Select an option (1-4): ").strip()
            if choice == "1":
                atm.check_balance()
            elif choice == "2":
                atm.deposit()
            elif choice == "3":
                atm.withdraw()
            elif choice == "4":
                print("Thank you for using ATM Simulator. Goodbye!")
                break
            else:
                raise InvalidOptionError("Invalid Menu Error: Option must be between 1 and 4!")
        except InvalidOptionError as err:
            print(err)

if __name__ == "__main__":
    main()
