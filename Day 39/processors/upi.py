# ==============================================================================
# Program    : UPI Payment Processor (upi.py)
# Objective  : Implement UPI payment processor satisfying PaymentProcessor protocol.
# Concept    : Structural Subtyping Concrete Implementation
# Why Used   : Handles UPI VPA payments.
# ==============================================================================

import os
import sys
import uuid

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from payment import Payment

class UPIPaymentProcessor:
    """UPI Payment Processor satisfying PaymentProcessor Protocol."""
    def __init__(self, vpa: str = "suraj@upi"):
        self.vpa = vpa

    def pay(self, payment: Payment) -> tuple[bool, str]:
        print(f"[UPI] Processing {payment.currency} {payment.amount:.2f} via VPA '{self.vpa}'")
        ref_id = f"UPI-{uuid.uuid4().hex[:8].upper()}"
        return True, ref_id
