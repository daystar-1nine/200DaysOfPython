# ==============================================================================
# Program    : Transaction Entity Dataclass (transaction.py)
# Objective  : Model Transaction lifecycle entity using @dataclass & default_factory.
# Concept    : Stateful Domain Entity Dataclass
# Why Used   : Tracks payment status, transaction ID, and timestamp dynamically.
# ==============================================================================

from dataclasses import dataclass, field
from datetime import datetime
import os
import sys
import uuid

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from enums import TransactionStatus
from payment import Payment

@dataclass
class Transaction:
    """Stateful Transaction domain entity."""
    payment: Payment
    # What is used : field(default_factory=...)
    # Why it is used: Generates unique UUID and current timestamp dynamically per instance
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TransactionStatus = TransactionStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    reference_id: str | None = None

    def mark_success(self, ref_id: str):
        self.status = TransactionStatus.SUCCESS
        self.reference_id = ref_id

    def mark_failed(self):
        self.status = TransactionStatus.FAILED
