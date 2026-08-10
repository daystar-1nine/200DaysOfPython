# ==============================================================================
# Program    : ATM Transaction Logger Demo (Task 2)
# Objective  : Log ATM transactions (INFO, WARNING, ERROR) to 'atm.log'.
# Concept    : Basic File-Based Logging Configuration
# Why Used   : Saves transaction audit trail into persistent log file.
# ==============================================================================

import logging
import os

log_file = "atm.log"

# What is used : logging.basicConfig() writing to atm.log
# Why it is used: Directs log events at INFO level and above into atm.log file
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)

class ATMSystem:
    def __init__(self, initial_balance=1000.0):
        self.balance = initial_balance
        logging.info("User logged in. Current Balance: Rs.%.2f", self.balance)

    def deposit(self, amount):
        if amount <= 0:
            logging.error("Invalid transaction: Deposit amount must be positive (Rs.%.2f)", amount)
            print("Error: Invalid deposit amount.")
            return
        self.balance += amount
        logging.info("Deposit successful: Added Rs.%.2f | New Balance: Rs.%.2f", amount, self.balance)
        print(f"Deposit successful. Balance: Rs.{self.balance:.2f}")

    def withdraw(self, amount):
        if amount <= 0:
            logging.error("Invalid transaction: Withdrawal amount must be positive (Rs.%.2f)", amount)
            print("Error: Invalid withdrawal amount.")
            return
        if amount > self.balance:
            logging.warning("Insufficient balance: Requested Rs.%.2f | Available Rs.%.2f", amount, self.balance)
            print("Warning: Insufficient balance.")
            return
        self.balance -= amount
        logging.info("Withdrawal successful: Debited Rs.%.2f | New Balance: Rs.%.2f", amount, self.balance)
        print(f"Withdrawal successful. Balance: Rs.{self.balance:.2f}")

def main():
    print("=== TASK 2: ATM LOGGER DEMO ===")
    atm = ATMSystem(1000.0)
    atm.deposit(500.0)
    atm.withdraw(200.0)
    atm.withdraw(2000.0)
    atm.deposit(-50.0)

    print(f"\nAll events logged to '{log_file}'. Previewing log content:")
    logging.shutdown()
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            print(f.read())
        os.remove(log_file)

if __name__ == "__main__":
    main()
