# ==============================================================================
# Program    : Bank Account Class & Unit Test Suite (Bonus Challenge)
# Objective  : Implement OOP BankAccount class and test transactions with setUp fixtures.
# Concept    : OOP Testing & Fixtures (setUp)
# Why Used   : Tests deposit, withdrawal, insufficient balance exceptions, and invalid inputs.
# ==============================================================================

import unittest

# --- BANK ACCOUNT CLASS ---
class BankAccount:
    def __init__(self, account_holder, initial_balance=0.0):
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        self.account_holder = account_holder
        self._balance = initial_balance

    def get_balance(self):
        return self._balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be strictly positive.")
        self._balance += amount
        return self._balance

    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be strictly positive.")
        if amount > self._balance:
            raise ValueError("Insufficient account balance for withdrawal.")
        self._balance -= amount
        return self._balance


# --- BANK ACCOUNT UNIT TEST SUITE ---
class TestBankAccount(unittest.TestCase):

    # What is used : setUp() fixture method
    # Why it is used: Initializes fresh BankAccount instance with Rs.1000 balance before each test method
    def setUp(self):
        self.account = BankAccount("Suraj Sawant", 1000.0)

    def test_initial_balance(self):
        self.assertEqual(self.account.get_balance(), 1000.0)

    def test_successful_deposit(self):
        new_balance = self.account.deposit(500.0)
        self.assertEqual(new_balance, 1500.0)
        self.assertEqual(self.account.get_balance(), 1500.0)

    def test_successful_withdrawal(self):
        new_balance = self.account.withdraw(400.0)
        self.assertEqual(new_balance, 600.0)
        self.assertEqual(self.account.get_balance(), 600.0)

    def test_insufficient_balance_withdrawal(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(2000.0)

    def test_negative_or_zero_deposit(self):
        with self.assertRaises(ValueError):
            self.account.deposit(-100.0)
        with self.assertRaises(ValueError):
            self.account.deposit(0.0)

    def test_negative_or_zero_withdrawal(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(-50.0)
        with self.assertRaises(ValueError):
            self.account.withdraw(0.0)

if __name__ == "__main__":
    unittest.main()
