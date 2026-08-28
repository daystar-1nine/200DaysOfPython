# ==============================================================================
# Program    : Bank Management System (Composition Layer)
# Objective  : Manage multiple accounts, transfers, interest application, and payments.
# Concept    : Composition ("Bank HAS-A Accounts") & Polymorphic Delegation
# Why Used   : Acts as centralized orchestrator for the banking system.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from account import Account, SavingsAccount, CurrentAccount
from payments import PaymentStrategy

class Bank:
    def __init__(self, name: str):
        self.name = name
        # What is used : Composition (self.accounts dictionary of Account objects)
        # Why it is used: Demonstrates HAS-A relationship where Bank maintains Account references
        self.accounts: dict[str, Account] = {}

    def create_savings_account(self, acc_num: str, holder_name: str, initial_balance: float = 1000.0) -> SavingsAccount:
        if acc_num in self.accounts:
            raise ValueError(f"Account number '{acc_num}' already exists.")
        acc = SavingsAccount(acc_num, holder_name, initial_balance)
        self.accounts[acc_num] = acc
        return acc

    def create_current_account(self, acc_num: str, holder_name: str, initial_balance: float = 0.0) -> CurrentAccount:
        if acc_num in self.accounts:
            raise ValueError(f"Account number '{acc_num}' already exists.")
        acc = CurrentAccount(acc_num, holder_name, initial_balance)
        self.accounts[acc_num] = acc
        return acc

    def get_account(self, acc_num: str) -> Account:
        if acc_num not in self.accounts:
            raise KeyError(f"Account number '{acc_num}' not found.")
        return self.accounts[acc_num]

    def process_payment(self, acc_num: str, amount: float, payment_strategy: PaymentStrategy) -> bool:
        account = self.get_account(acc_num)
        if account.withdraw(amount):
            return payment_strategy.pay(amount)
        return False

    def apply_monthly_interest(self) -> dict[str, float]:
        applied = {}
        for acc_num, acc in self.accounts.items():
            interest = acc.calculate_interest()
            if interest > 0:
                acc.deposit(interest)
                applied[acc_num] = interest
        return applied
