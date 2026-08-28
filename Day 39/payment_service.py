# ==============================================================================
# Program    : Payment Service Layer (payment_service.py)
# Objective  : Orchestrate payment processing using PaymentProcessor protocol interface.
# Concept    : Protocol Composition & Service Layer
# Why Used   : Decouples payment execution from specific gateway implementations.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

models_dir = os.path.join(pkg_root, "models")
processors_dir = os.path.join(pkg_root, "processors")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
if processors_dir not in sys.path:
    sys.path.insert(0, processors_dir)

from models.enums import PaymentMethod, TransactionStatus
from models.payment import Payment
from models.transaction import Transaction
from processors.protocol import PaymentProcessor
from processors.upi import UPIPaymentProcessor
from processors.card import CardPaymentProcessor
from processors.bank import BankTransferPaymentProcessor
from processors.wallet import WalletPaymentProcessor

class PaymentService:
    def __init__(self):
        # Register default processor strategy map
        self.processors: dict[PaymentMethod, PaymentProcessor] = {
            PaymentMethod.UPI: UPIPaymentProcessor(),
            PaymentMethod.CARD: CardPaymentProcessor(),
            PaymentMethod.BANK_TRANSFER: BankTransferPaymentProcessor(),
            PaymentMethod.WALLET: WalletPaymentProcessor()
        }

    def process_payment(self, amount: float, method: PaymentMethod, description: str = "") -> Transaction:
        payment = Payment(amount=amount, method=method, description=description)
        transaction = Transaction(payment=payment)

        processor = self.processors.get(method)
        if not processor:
            transaction.mark_failed()
            raise ValueError(f"No processor registered for payment method: {method}")

        success, ref_id = processor.pay(payment)
        if success:
            transaction.mark_success(ref_id)
        else:
            transaction.mark_failed()

        return transaction


if __name__ == "__main__":
    print("==================================================")
    print("      DAY 39 - PAYMENT DOMAIN MODEL SYSTEM       ")
    print("==================================================\n")

    service = PaymentService()

    tx1 = service.process_payment(1250.0, PaymentMethod.UPI, "Dinner Bill")
    print(f"Transaction ID: {tx1.id} | Status: {tx1.status.value} | Ref: {tx1.reference_id}\n")

    tx2 = service.process_payment(4500.0, PaymentMethod.CARD, "Laptop Accessory")
    print(f"Transaction ID: {tx2.id} | Status: {tx2.status.value} | Ref: {tx2.reference_id}\n")
