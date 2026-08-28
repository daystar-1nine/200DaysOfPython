# ==============================================================================
# Program    : Payment Strategy Implementations (Bonus Feature)
# Objective  : Provide abstract PaymentStrategy and concrete UPI, Card, and Transfer strategies.
# Concept    : Polymorphism & Strategy Pattern
# Why Used   : Allows seamless extension of new payment methods without modifying core banking logic.
# ==============================================================================

from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        """Abstract payment method."""
        pass

class UPIPayment(PaymentStrategy):
    def __init__(self, upi_id: str):
        self.upi_id = upi_id

    def pay(self, amount: float) -> bool:
        print(f"  [UPI PAYMENT] Processing Rs.{amount:,.2f} via UPI ID '{self.upi_id}'... SUCCESS!")
        return True

class CardPayment(PaymentStrategy):
    def __init__(self, card_number: str, card_holder: str):
        self.card_number = card_number
        self.card_holder = card_holder

    def pay(self, amount: float) -> bool:
        masked = "*" * 12 + self.card_number[-4:] if len(self.card_number) >= 4 else self.card_number
        print(f"  [CARD PAYMENT] Processing Rs.{amount:,.2f} via Card '{masked}' ({self.card_holder})... SUCCESS!")
        return True

class BankTransferPayment(PaymentStrategy):
    def __init__(self, ifsc_code: str, target_acc_num: str):
        self.ifsc_code = ifsc_code
        self.target_acc_num = target_acc_num

    def pay(self, amount: float) -> bool:
        print(f"  [BANK TRANSFER] Processing Rs.{amount:,.2f} to Account '{self.target_acc_num}' (IFSC: {self.ifsc_code})... SUCCESS!")
        return True
