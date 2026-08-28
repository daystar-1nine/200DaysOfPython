# ==============================================================================
# Program    : Payment Processor Protocol Interface (protocol.py)
# Objective  : Define PaymentProcessor protocol contract using typing.Protocol.
# Concept    : Structural Subtyping (Duck Typing with Type Checking)
# Why Used   : Allows any class satisfying pay() method to act as a payment gateway.
# ==============================================================================

from typing import Protocol, runtime_checkable
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from payment import Payment

@runtime_checkable
class PaymentProcessor(Protocol):
    """Protocol interface defining required payment processor methods."""
    # What is used : typing.Protocol structural typing contract
    # Why it is used: Enforces pay() method signature without requiring explicit inheritance
    def pay(self, payment: Payment) -> tuple[bool, str]:
        """Process payment and return tuple of (success_boolean, reference_id)."""
        ...
