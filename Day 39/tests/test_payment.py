# ==============================================================================
# Test Suite : Day 39 Payment Domain Model Pytest Suite (16 Test Cases)
# Objective  : Test Dataclasses, Enums, Protocols, and Payment Service execution.
# Concept    : Unit Testing Dataclasses & Structural Subtyping Protocols
# Why Used   : Asserts domain immutability, protocol compliance, and state transitions.
# ==============================================================================

from dataclasses import FrozenInstanceError
import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from payment_service import PaymentService

# 1. Test Valid Payment Dataclass
def test_valid_payment_dataclass():
    p = Payment(amount=500.0, method=PaymentMethod.UPI)
    assert p.amount == 500.0
    assert p.method == PaymentMethod.UPI
    assert p.currency == "INR"

# 2. Test Invalid Amount Raises ValueError
def test_invalid_payment_amount():
    with pytest.raises(ValueError, match="greater than zero"):
        Payment(amount=-100.0, method=PaymentMethod.CARD)

# 3. Test Payment Immutability (frozen=True)
def test_payment_dataclass_immutability():
    p = Payment(amount=100.0, method=PaymentMethod.UPI)
    with pytest.raises(FrozenInstanceError):
        p.amount = 200.0  # type: ignore

# 4. Test Transaction Default UUID & Timestamp
def test_transaction_defaults():
    p = Payment(amount=250.0, method=PaymentMethod.WALLET)
    tx = Transaction(payment=p)
    assert tx.id is not None
    assert tx.status == TransactionStatus.PENDING
    assert tx.reference_id is None

# 5. Test Transaction Status Success Transition
def test_transaction_mark_success():
    p = Payment(amount=300.0, method=PaymentMethod.CARD)
    tx = Transaction(payment=p)
    tx.mark_success("REF123")
    assert tx.status == TransactionStatus.SUCCESS
    assert tx.reference_id == "REF123"

# 6. Test Transaction Status Failed Transition
def test_transaction_mark_failed():
    p = Payment(amount=300.0, method=PaymentMethod.CARD)
    tx = Transaction(payment=p)
    tx.mark_failed()
    assert tx.status == TransactionStatus.FAILED

# 7. Test UPI Processor Protocol Compliance
def test_upi_processor_protocol():
    proc = UPIPaymentProcessor()
    assert isinstance(proc, PaymentProcessor)
    p = Payment(amount=100.0, method=PaymentMethod.UPI)
    ok, ref = proc.pay(p)
    assert ok is True
    assert ref.startswith("UPI-")

# 8. Test Card Processor Protocol Compliance
def test_card_processor_protocol():
    proc = CardPaymentProcessor()
    assert isinstance(proc, PaymentProcessor)
    p = Payment(amount=200.0, method=PaymentMethod.CARD)
    ok, ref = proc.pay(p)
    assert ok is True
    assert ref.startswith("CARD-")

# 9. Test Bank Processor Protocol Compliance
def test_bank_processor_protocol():
    proc = BankTransferPaymentProcessor()
    assert isinstance(proc, PaymentProcessor)
    p = Payment(amount=1500.0, method=PaymentMethod.BANK_TRANSFER)
    ok, ref = proc.pay(p)
    assert ok is True
    assert ref.startswith("BANK-")

# 10. Test Wallet Processor Protocol Compliance
def test_wallet_processor_protocol():
    proc = WalletPaymentProcessor()
    assert isinstance(proc, PaymentProcessor)
    p = Payment(amount=50.0, method=PaymentMethod.WALLET)
    ok, ref = proc.pay(p)
    assert ok is True
    assert ref.startswith("WAL-")

# 11. Test PaymentService UPI Execution
def test_payment_service_upi():
    service = PaymentService()
    tx = service.process_payment(100.0, PaymentMethod.UPI)
    assert tx.status == TransactionStatus.SUCCESS
    assert tx.reference_id is not None

# 12. Test PaymentService Card Execution
def test_payment_service_card():
    service = PaymentService()
    tx = service.process_payment(500.0, PaymentMethod.CARD)
    assert tx.status == TransactionStatus.SUCCESS

# 13. Test PaymentService Bank Execution
def test_payment_service_bank():
    service = PaymentService()
    tx = service.process_payment(2000.0, PaymentMethod.BANK_TRANSFER)
    assert tx.status == TransactionStatus.SUCCESS

# 14. Test PaymentService Wallet Execution
def test_payment_service_wallet():
    service = PaymentService()
    tx = service.process_payment(75.0, PaymentMethod.WALLET)
    assert tx.status == TransactionStatus.SUCCESS

# 15. Test Enum Safety
def test_enum_safety():
    assert PaymentMethod.UPI.value == "UPI"
    assert TransactionStatus.SUCCESS.value == "SUCCESS"

class TestPaymentRunner(unittest.TestCase):
    def test_payment_standalone(self):
        p = Payment(amount=10.0, method=PaymentMethod.UPI)
        self.assertEqual(p.amount, 10.0)

if __name__ == "__main__":
    unittest.main()
