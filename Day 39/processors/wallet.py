# ==============================================================================
# Program    : Wallet Payment Processor (wallet.py)
# Objective  : Implement Wallet payment processor satisfying PaymentProcessor protocol.
# Concept    : Structural Subtyping Concrete Implementation
# Why Used   : Handles Digital Wallet payments.
# ==============================================================================

import os
import sys
import uuid

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from payment import Payment

class WalletPaymentProcessor:
    """Wallet Payment Processor satisfying PaymentProcessor Protocol."""
    def __init__(self, wallet_id: str = "WAL-8899"):
        self.wallet_id = wallet_id

    def pay(self, payment: Payment) -> tuple[bool, str]:
        print(f"[WALLET] Deducting {payment.currency} {payment.amount:.2f} from Wallet '{self.wallet_id}'")
        ref_id = f"WAL-{uuid.uuid4().hex[:8].upper()}"
        return True, ref_id
