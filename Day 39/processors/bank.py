# ==============================================================================
# Program    : Bank Transfer Payment Processor (bank.py)
# Objective  : Implement Bank Transfer payment processor satisfying PaymentProcessor protocol.
# Concept    : Structural Subtyping Concrete Implementation
# Why Used   : Handles Direct NEFT/IMPS Bank Transfer payments.
# ==============================================================================

import os
import sys
import uuid

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from payment import Payment

class BankTransferPaymentProcessor:
    """Bank Transfer Payment Processor satisfying PaymentProcessor Protocol."""
    def __init__(self, account_no: str = "9876543210", ifsc: str = "SBIN0001234"):
        self.account_no = account_no
        self.ifsc = ifsc

    def pay(self, payment: Payment) -> tuple[bool, str]:
        print(f"[BANK] Transferring {payment.currency} {payment.amount:.2f} to Account '{self.account_no}' (IFSC: {self.ifsc})")
        ref_id = f"BANK-{uuid.uuid4().hex[:8].upper()}"
        return True, ref_id
