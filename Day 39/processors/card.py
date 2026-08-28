# ==============================================================================
# Program    : Card Payment Processor (card.py)
# Objective  : Implement Card payment processor satisfying PaymentProcessor protocol.
# Concept    : Structural Subtyping Concrete Implementation
# Why Used   : Handles Credit & Debit Card payments.
# ==============================================================================

import os
import sys
import uuid

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from payment import Payment

class CardPaymentProcessor:
    """Card Payment Processor satisfying PaymentProcessor Protocol."""
    def __init__(self, card_number_last4: str = "4321"):
        self.card_number_last4 = card_number_last4

    def pay(self, payment: Payment) -> tuple[bool, str]:
        print(f"[CARD] Processing {payment.currency} {payment.amount:.2f} via Card ending in '*{self.card_number_last4}'")
        ref_id = f"CARD-{uuid.uuid4().hex[:8].upper()}"
        return True, ref_id
