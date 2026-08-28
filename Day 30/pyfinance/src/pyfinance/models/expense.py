# ==============================================================================
# Program    : PyFinance Domain Models
# Objective  : Dataclasses representing Expense, Budget, and CurrencyRate entities.
# Concept    : Type-Safe Data Models with Dataclasses
# Why Used   : Encapsulates domain entity state instead of loose dictionaries.
# ==============================================================================

from dataclasses import dataclass
from datetime import datetime

@dataclass
class Expense:
    id: int | None
    amount: float
    category: str
    description: str
    date: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class Budget:
    id: int | None
    category: str
    monthly_limit: float

@dataclass
class CurrencyRate:
    base: str
    target: str
    rate: float
    updated_at: str
