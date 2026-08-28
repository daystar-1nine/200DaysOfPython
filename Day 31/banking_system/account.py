# ==============================================================================
# Program    : Banking System Domain Models (Abstract & Concrete Accounts)
# Objective  : Implement Account(ABC), SavingsAccount, and CurrentAccount using OOP.
# Concept    : Advanced OOP: ABCs, Inheritance, Polymorphism, Super, and Encapsulation
# Why Used   : Provides clean domain object hierarchy for banking application.
# ==============================================================================

from abc import ABC, abstractmethod
from datetime import datetime

class Account(ABC):
    def __init__(self, account_number: str, holder_name: str, initial_balance: float = 0.0):
        # What is used : Encapsulated Private Attributes & Transaction Log
        # Why it is used: Protects state from illegal direct mutation and tracks history
        if initial_balance < 0:
            raise ValueError("Initial balance cannot be negative.")
        self.account_number = account_number
        self.holder_name = holder_name
        self._balance = initial_balance
        self._transactions: list[dict] = []
        self._log_transaction("INITIAL_DEPOSIT", initial_balance, "Account opened")

    @property
    def balance(self) -> float:
        return self._balance

    def _log_transaction(self, tx_type: str, amount: float, description: str) -> None:
        self._transactions.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": tx_type,
            "amount": amount,
            "balance_after": self._balance,
            "description": description
        })

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be greater than zero.")
        self._balance += amount
        self._log_transaction("DEPOSIT", amount, "Deposit into account")

    @abstractmethod
    def withdraw(self, amount: float) -> bool:
        """Abstract method forcing concrete subclasses to define custom withdrawal rules."""
        pass

    @abstractmethod
    def calculate_interest(self) -> float:
        """Abstract method for computing account interest."""
        pass

    def transfer(self, target_account: "Account", amount: float) -> bool:
        if amount <= 0:
            raise ValueError("Transfer amount must be greater than zero.")
        if self.withdraw(amount):
            target_account.deposit(amount)
            self._log_transaction("TRANSFER_OUT", amount, f"Transfer to {target_account.account_number}")
            target_account._log_transaction("TRANSFER_IN", amount, f"Transfer from {self.account_number}")
            return True
        return False

    def get_transaction_history(self) -> list[dict]:
        return list(self._transactions)

class SavingsAccount(Account):
    def __init__(self, account_number: str, holder_name: str, initial_balance: float = 1000.0, interest_rate: float = 0.04, min_balance: float = 500.0):
        # What is used : super().__init__() constructor chaining
        # Why it is used: Initializes base Account attributes while setting Savings specific rules
        super().__init__(account_number, holder_name, initial_balance)
        self.interest_rate = interest_rate
        self.min_balance = min_balance

    def withdraw(self, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be greater than zero.")
        if self._balance - amount < self.min_balance:
            raise ValueError(f"Withdrawal denied. Must maintain minimum balance of Rs.{self.min_balance:.2f}")
        self._balance -= amount
        self._log_transaction("WITHDRAWAL", amount, "Savings withdrawal")
        return True

    def calculate_interest(self) -> float:
        interest = self._balance * self.interest_rate
        return interest

class CurrentAccount(Account):
    def __init__(self, account_number: str, holder_name: str, initial_balance: float = 0.0, overdraft_limit: float = 5000.0):
        super().__init__(account_number, holder_name, initial_balance)
        self.overdraft_limit = overdraft_limit

    def withdraw(self, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be greater than zero.")
        if self._balance - amount < -self.overdraft_limit:
            raise ValueError(f"Withdrawal denied. Exceeds overdraft limit of Rs.{self.overdraft_limit:.2f}")
        self._balance -= amount
        self._log_transaction("WITHDRAWAL", amount, "Current account withdrawal")
        return True

    def calculate_interest(self) -> float:
        # Current accounts earn zero interest
        return 0.0
