# ==============================================================================
# Program    : Payment Value Object Dataclass (payment.py)
# Objective  : Model immutable Payment details using @dataclass with frozen=True & slots=True.
# Concept    : Immutable Dataclass Value Object
# Why Used   : Represents payment monetary details safely without accidental mutations.
# ==============================================================================

from dataclasses import dataclass
import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from enums import PaymentMethod

@dataclass(frozen=True, slots=True)
class Payment:
    """Immutable Payment dataclass value object."""
    # What is used : @dataclass(frozen=True, slots=True)
    # Why it is used: Enforces immutability and reduces RAM allocation footprint
    amount: float
    method: PaymentMethod
    currency: str = "INR"
    description: str = "Payment Transaction"

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("Payment amount must be greater than zero.")
