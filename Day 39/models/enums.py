# ==============================================================================
# Program    : Payment & Transaction Enums (enums.py)
# Objective  : Define PaymentMethod and TransactionStatus enumeration constants.
# Concept    : Type-Safe Enumerations
# Why Used   : Eliminates raw string typos and enforces valid domain states.
# ==============================================================================

from enum import Enum

class PaymentMethod(Enum):
    """Enumeration of supported payment processors."""
    # What is used : String Enum Constants
    # Why it is used: Provides type-safe payment gateway identification
    UPI = "UPI"
    CARD = "CARD"
    BANK_TRANSFER = "BANK_TRANSFER"
    WALLET = "WALLET"

class TransactionStatus(Enum):
    """Enumeration of transaction lifecycle states."""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
