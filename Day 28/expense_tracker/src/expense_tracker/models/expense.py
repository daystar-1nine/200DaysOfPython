# ==============================================================================
# Program    : Domain Models (Expense, User, Category)
# Objective  : Dataclasses representing core domain objects.
# Concept    : Dataclasses & Data Modeling
# Why Used   : Provides type safety instead of passing raw tuples or dicts.
# ==============================================================================

from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

@dataclass
class Category:
    id: int
    name: str

@dataclass
class Expense:
    id: int
    user_id: int
    category_id: int
    amount: float
    description: str
    date: str
