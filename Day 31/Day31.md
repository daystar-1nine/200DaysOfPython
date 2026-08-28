# 🐍 Day 31/200 – Masterclass Notes: Advanced OOP & Banking System

🎯 **Goal:** Transition from basic class definitions to designing software architectures with **Advanced Object-Oriented Programming (OOP)** in Python—mastering Inheritance, Polymorphism, Abstract Base Classes (`ABC`, `@abstractmethod`), Composition ("HAS-A") vs. Inheritance ("IS-A"), Encapsulation with `@property` decorators, Class (`@classmethod`) vs. Static (`@staticmethod`) methods, and an introduction to **SOLID Principles**.

---

## 📌 Executive Summary & Key Takeaways

- **Inheritance ("IS-A" relationship):** Allows a child class (`SavingsAccount`) to inherit attributes and methods from a parent class (`Account`) while extending or overriding behaviors using `super()`.
- **Abstract Base Classes (`ABC` & `@abstractmethod`):** Force child subclasses to implement required interface contracts (`withdraw`, `calculate_interest`), raising `TypeError` at instantiation if any abstract methods are missing.
- **Polymorphism:** Enables treating different concrete objects (`SavingsAccount`, `CurrentAccount`, `UPIPayment`, `CardPayment`) through a unified interface.
- **Composition ("HAS-A" relationship):** Favor composition over inheritance (`Bank` HAS-A collection of `Account` instances).
- **Encapsulation & `@property` Decorators:** Protect internal state (`_balance`) from illegal direct mutations while offering clean attribute access syntax (`account.balance`).
- **SOLID Principles:**
  - **S**ingle Responsibility Principle: A class should have one reason to change.
  - **O**pen/Closed Principle: Open for extension, closed for modification.
  - **L**iskov Substitution Principle: Subtypes must be substitutable for base types.
  - **I**nterface Segregation Principle: Prefer targeted interfaces over fat base classes.
  - **D**ependency Inversion Principle: Depend upon abstractions, not concrete implementations.

---

## 📖 Topic 1: Abstract Base Classes (`ABC`) & Inheritance

```python
from abc import ABC, abstractmethod

class Account(ABC):
    def __init__(self, account_number: str, holder_name: str, initial_balance: float):
        self.account_number = account_number
        self.holder_name = holder_name
        self._balance = initial_balance  # Encapsulated state

    @property
    def balance(self) -> float:
        return self._balance

    @abstractmethod
    def withdraw(self, amount: float) -> bool:
        """Abstract method forcing concrete subclasses to define withdrawal logic."""
        pass
```

---

## 📖 Topic 2: Polymorphism & Strategy Pattern

```python
class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        pass

class UPIPayment(PaymentStrategy):
    def __init__(self, upi_id: str):
        self.upi_id = upi_id

    def pay(self, amount: float) -> bool:
        print(f"Processing UPI payment of Rs.{amount:.2f} via {self.upi_id}")
        return True

class CardPayment(PaymentStrategy):
    def __init__(self, card_number: str):
        self.card_number = card_number

    def pay(self, amount: float) -> bool:
        print(f"Processing Card payment of Rs.{amount:.2f} via Card ending in {self.card_number[-4:]}")
        return True
```

---

## ⚡ Master Cheat Sheet

```python
# Advanced OOP Master Cheat Sheet

# 1. Calling Superclass Initializer
super().__init__(account_number, holder_name, balance)

# 2. Defining Encapsulated Property
@property
def balance(self) -> float:
    return self._balance

# 3. Method Types
@classmethod
def from_string(cls, data_str: str):  # Alternative constructor
    pass

@staticmethod
def is_valid_account_number(acc_num: str) -> bool:  # Pure utility method
    return len(acc_num) == 10 and acc_num.isdigit()
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forcing Deep Inheritance Trees Instead of Composition:**
   - ❌ Creating `Account` $\rightarrow$ `BankUserAccount` $\rightarrow$ `PremiumBankUserAccount` $\rightarrow$ `VIPPremiumBankUserAccount`.
   - ✅ Prefer Composition (`Bank` HAS-A `Account`, `Account` HAS-A `Customer`).

2. **Directly Mutating Private Attributes (`_balance`):**
   - ❌ `account._balance -= 5000` (bypasses validation rules).
   - ✅ Use proper deposit/withdraw methods that enforce business invariants.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What happens if a subclass does not implement an `@abstractmethod`?
**Answer:** Python will raise a `TypeError: Can't instantiate abstract class SubClass with abstract method...` at runtime as soon as you attempt to create an instance of the subclass.

### Q2: What is the difference between `@classmethod` and `@staticmethod`?
**Answer:** `@classmethod` receives the class (`cls`) as its first implicit parameter and can modify class state or act as alternative constructors. `@staticmethod` receives no implicit class/self parameter and acts as a pure utility function scoped under the class namespace.

---

## 📝 Recap Checklist
- [x] Implemented abstract base class `Account(ABC)` with `@abstractmethod`.
- [x] Created concrete `SavingsAccount` and `CurrentAccount` subclasses overriding methods.
- [x] Used `super()` to chain constructor calls.
- [x] Implemented `@property` getters for encapsulated attributes.
- [x] Built Strategy Pattern payment options (`UPIPayment`, `CardPayment`, `BankTransferPayment`).
- [x] Implemented composition in `Bank` class managing accounts and transactions.
- [x] Built unit and integration tests covering OOP banking operations.
