# 🐍 Day 39/200 – Masterclass Notes: Dataclasses, Enums, Protocols & Static Type Checking

🎯 **Goal:** Master modern Python domain modeling tools—understanding `@dataclass` (with `field(default_factory=...)`, `frozen=True`, `slots=True`), `Enum` type-safe enumerations, `typing.Protocol` structural subtyping, and static type checking with `mypy` by building a production-grade **Payment Domain Model** supporting UPI, Card, Bank Transfer, and Digital Wallet payment processors.

---

## 📌 Executive Summary & Key Takeaways

- **Dataclasses (`@dataclass`):** Automatically generates boilerplate methods (`__init__`, `__repr__`, `__eq__`). `frozen=True` creates immutable value objects; `slots=True` lowers memory overhead by suppressing `__dict__`.
- **Field Default Factories:** Use `field(default_factory=list)` or `field(default_factory=datetime.now)` for mutable default values to prevent shared default state bugs across instances.
- **Enums (`Enum`):** Replaces error-prone raw strings (`"pending"`) with type-safe enumeration constants (`TransactionStatus.PENDING`), catching typos at compile-time/type-check time.
- **Protocols (`typing.Protocol`):** Enables **Structural Subtyping** (duck typing with type checking). A class satisfies a `Protocol` simply by implementing the required methods—no explicit inheritance required!
- **Static Type Checking (`mypy`):** Analyzes type hints across codebases without running Python code, preventing `TypeError` and `AttributeError` bugs in production.

---

## 📖 Topic 1: Modern Dataclasses & Enums

```python
from dataclasses import dataclass, field
from enum import Enum
import uuid

class TransactionStatus(Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

@dataclass(frozen=True, slots=True)
class Payment:
    amount: float
    currency: str = "INR"

@dataclass
class Transaction:
    payment: Payment
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TransactionStatus = TransactionStatus.PENDING
```

---

## 📖 Topic 2: Protocols & Structural Subtyping

```python
from typing import Protocol

class PaymentProcessor(Protocol):
    """Protocol contract defining required payment processor methods."""
    def pay(self, payment: Payment) -> bool:
        ...

# Concrete implementation satisfying PaymentProcessor without inheriting from it
class UPIProcessor:
    def pay(self, payment: Payment) -> bool:
        print(f"Processing UPI payment of Rs.{payment.amount}")
        return True
```

---

## ⚡ Master Cheat Sheet

```python
# Dataclasses, Enums & Protocols Cheat Sheet

# 1. Immutable Dataclass with Slots
@dataclass(frozen=True, slots=True)
class User:
    id: int
    name: str

# 2. Enum Definition
class PaymentMethod(Enum):
    UPI = "UPI"
    CARD = "CARD"
    BANK_TRANSFER = "BANK_TRANSFER"
    WALLET = "WALLET"

# 3. Protocol Definition
class Processor(Protocol):
    def process(self, data: str) -> bool:
        ...
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Using Mutable Defaults in `@dataclass`:**
   - ❌ Defining `tags: list = []` causes every class instance to share the exact same list object in RAM.
   - ✅ Always use `tags: list = field(default_factory=list)`.

2. **Expecting `Protocol` to Enforce Runtime Checks Without Method Calls:**
   - ❌ Protocols are static type checking contracts analyzed by `mypy`. Python's interpreter does not enforce protocol methods unless called at runtime or checked via `isinstance` with `@runtime_checkable`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between nominal subtyping (Abstract Base Classes) and structural subtyping (Protocols)?
**Answer:** Nominal subtyping (ABC) requires explicit class inheritance (`class UPI(ABC)`). Structural subtyping (Protocol) requires only that the target class implements the matching signature and methods—no explicit inheritance is necessary.

### Q2: Why use `slots=True` in dataclasses?
**Answer:** `slots=True` instructs Python not to create a dynamic `__dict__` for instances, significantly reducing memory footprint and speeding up attribute access.

---

## 📝 Recap Checklist
- [x] Mastered `@dataclass` with `default_factory`, `frozen=True`, and `slots=True`.
- [x] Implemented `PaymentMethod` and `TransactionStatus` Enums.
- [x] Defined `PaymentProcessor` Protocol interface.
- [x] Created concrete processors: `UPIPaymentProcessor`, `CardPaymentProcessor`, `BankTransferPaymentProcessor`, `WalletPaymentProcessor`.
- [x] Created Pytest test suite testing valid transactions, invalid amounts, status transitions, and all 4 processors (15+ test cases).
