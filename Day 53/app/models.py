"""
===============================================================================
DAY 53 — DATA MODELS MODULE
===============================================================================
This module defines the Sale domain entity using a dataclass with computed
derived property total and serialization helper methods.
===============================================================================
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Any


@dataclass
class Sale:
    """Dataclass representing a validated sales transaction record."""
    order_id: int
    customer: str
    product: str
    category: str
    price: float
    quantity: int
    date: date

    @property
    def total(self) -> float:
        """Calculate derived total order amount dynamically."""
        # What is used: Dataclass property decorator.
        # Why it is used: Provides on-the-fly total calculation without mutating state.
        # How it works: Multiplies unit price float by quantity integer.
        return self.price * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Sale instance into dictionary structure."""
        # What is used: Standard dictionary mapping.
        # Why it is used: Facilitates writing cleaned records to CSV format.
        # How it works: Converts attributes and date object into standard types.
        return {
            "order_id": self.order_id,
            "customer": self.customer,
            "product": self.product,
            "category": self.category,
            "price": self.price,
            "quantity": self.quantity,
            "date": self.date.strftime("%Y-%m-%d"),
            "total": self.total,
        }
