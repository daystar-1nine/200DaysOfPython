# ==============================================================================
# Program    : ATM Banking System with Audit Logging (Challenge Project)
# Objective  : Build complete ATM Banking app logging login attempts, deposits, withdrawals, and errors to 'atm.log'.
# Concept    : Full Application Event & Audit Trail Logging
# Why Used   : Captures INFO, WARNING, and ERROR logs with exact timestamps and account balances.
# ==============================================================================

import logging
import os

log_file = os.path.join(os.path.dirname(__file__), "atm.log")

# Setup ATM logger
logger = logging.getLogger("ATMSystem")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

class ATMBank:
    def __init__(self, pin="1234", balance=10000.0):
        self.correct_pin = pin
        self.balance = balance
        self.authenticated = False

    def login(self, entered_pin):
        logger.info("Login attempt with PIN: %s...", entered_pin[:2] + "**")
        if entered_pin == self.correct_pin:
            self.authenticated = True
            logger.info("Login successful. Initial Balance: Rs.%.2f", self.balance)
            print("Login Successful! Welcome to ATM Services.")
            return True
        else:
            logger.warning("Invalid PIN attempt! Authentication failed.")
            print("Error: Invalid PIN entered.")
            return False

    def deposit(self, amount):
        if not self.authenticated:
            logger.error("Unauthorized deposit attempt without login!")
            print("Error: Please login first.")
            return
        if amount <= 0:
            logger.error("Invalid transaction: Deposit amount must be positive (Rs.%.2f)", amount)
            print("Error: Invalid deposit amount.")
            return
        self.balance += amount
        logger.info("Deposit: Added Rs.%.2f | New Balance: Rs.%.2f", amount, self.balance)
        print(f"Deposit Successful! Current Balance: Rs.{self.balance:,.2f}")

    def withdraw(self, amount):
        if not self.authenticated:
            logger.error("Unauthorized withdrawal attempt without login!")
            print("Error: Please login first.")
            return
        if amount <= 0:
            logger.error("Invalid transaction: Withdrawal amount must be positive (Rs.%.2f)", amount)
            print("Error: Invalid withdrawal amount.")
            return
        if amount > self.balance:
            logger.warning("Insufficient Balance: Requested Rs.%.2f | Available Rs.%.2f", amount, self.balance)
            print("Error: Insufficient account balance.")
            return
        self.balance -= amount
        logger.info("Withdrawal: Debited Rs.%.2f | New Balance: Rs.%.2f", amount, self.balance)
        print(f"Withdrawal Successful! Current Balance: Rs.{self.balance:,.2f}")

def main():
    print("==========================================================")
    print("            ATM BANKING SYSTEM WITH LOGGING               ")
    print("==========================================================")
    
    atm = ATMBank(pin="1234", balance=10000.0)

    # 1. Invalid Login Attempt
    atm.login("9999")

    # 2. Valid Login Attempt
    if atm.login("1234"):
        atm.deposit(5000.0)
        atm.withdraw(1000.0)
        atm.withdraw(20000.0)  # Insufficient balance
        atm.deposit(-500.0)    # Invalid transaction

    print(f"\nAudit log created at '{log_file}'. Exiting ATM System.")

if __name__ == "__main__":
    main()
