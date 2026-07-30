# 🐍 Day 13/200 – Masterclass Notes: Object-Oriented Programming (OOP) – Part 2

🎯 **Goal:** Master the four fundamental pillars of Object-Oriented Programming—**Inheritance**, **Polymorphism**, **Encapsulation**, and **Abstraction**—along with Method Overriding, `super()`, Getters/Setters, Name Mangling, and Abstract Base Classes (`ABC`).

---

## 📌 Executive Summary & Key Takeaways

- **The Four Pillars of OOP:**
  1. **Inheritance:** Allows child classes to inherit attributes and methods from parent classes to eliminate code duplication.
  2. **Polymorphism ("Many Forms"):** Allows different object types to respond to the same method invocation interface.
  3. **Encapsulation:** Restricts direct access to an object's internal state (using private `__attributes`) and exposes safe accessor/mutator methods (Getters/Setters).
  4. **Abstraction:** Hides complex underlying implementation details and exposes only essential features via Abstract Base Classes (`ABC`).
- **`super()` Function:** Delegates attribute and method initialization up the parent class hierarchy.
- **Method Resolution Order (MRO):** Python's C3 Linearization algorithm that determines the order in which base classes are searched during multiple inheritance lookup.

---

## 📖 Topic 1: Inheritance & Types of Inheritance

### 1.1 Inheritance Hierarchies

```python
# 1. Single Inheritance
class Parent: pass
class Child(Parent): pass

# 2. Multilevel Inheritance
class Grandparent: pass
class Parent(Grandparent): pass
class Child(Parent): pass

# 3. Multiple Inheritance
class Father: pass
class Mother: pass
class Child(Father, Mother): pass

# 4. Hierarchical Inheritance
class Animal: pass
class Dog(Animal): pass
class Cat(Animal): pass
```

---

## 📖 Topic 2: Method Overriding & `super()`

### 2.1 Overriding Parent Implementation

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def make_sound(self):
        return "Generic animal sound"

class Dog(Animal):
    def __init__(self, name, breed):
        # Delegate name initialization to parent constructor
        super().__init__(name)
        self.breed = breed

    # Override parent method
    def make_sound(self):
        return "Woof! Woof!"
```

---

## 📖 Topic 3: Polymorphism ("One Interface, Many Forms")

Polymorphism allows functions to process objects of different classes uniformly as long as they implement a common method interface (Duck Typing: *"If it walks like a duck and quacks like a duck, it's a duck"*).

```python
class Dog:
    def speak(self): return "Bark"

class Cat:
    def speak(self): return "Meow"

class Duck:
    def speak(self): return "Quack"

# Heterogeneous collection processed uniformly
animals = [Dog(), Cat(), Duck()]

for a in animals:
    print(a.speak())  # Polymorphic method call
```

---

## 📖 Topic 4: Encapsulation (Access Modifiers & Getters/Setters)

Python uses naming conventions for access control:

| Access Type | Syntax Prefix | Accessibility | Example |
|---|---|---|---|
| **Public** | None | Accessible everywhere | `self.name` |
| **Protected** | `_` (Single underscore) | Convention: Internal use within class & subclasses | `self._department` |
| **Private** | `__` (Double underscore) | Name-mangled: Hidden from direct external access | `self.__balance` |

```python
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self.__balance = balance  # Private attribute

    # Getter (Accessor)
    def get_balance(self):
        return self.__balance

    # Setter (Mutator) with validation
    def set_balance(self, amount):
        if amount >= 0:
            self.__balance = amount
        else:
            raise ValueError("Balance cannot be negative!")
```

---

## 📖 Topic 5: Abstraction & Abstract Base Classes (`ABC`)

Abstract classes cannot be instantiated directly. They enforce a contract requiring all subclasses to implement decorated `@abstractmethod` functions.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def calculate_area(self):
        """Abstract method - must be overridden by subclasses."""
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    # Implementing required abstract method
    def calculate_area(self):
        import math
        return math.pi * (self.radius ** 2)
```

---

## ⚡ Master Cheat Sheet

```python
# Advanced OOP Cheat Sheet

from abc import ABC, abstractmethod

# Abstract Base Class
class Vehicle(ABC):
    def __init__(self, brand):
        self.brand = brand
        self.__engine_started = False  # Private variable

    @abstractmethod
    def start_engine(self): pass

# Child Class implementing interface
class Car(Vehicle):
    def start_engine(self):
        return f"{self.brand} car engine started!"

# Polymorphism Loop
vehicles = [Car("Toyota"), Car("Honda")]
for v in vehicles:
    print(v.start_engine())
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Trying to Instantiate Abstract Classes:**
   - ❌ `shape = Shape()` (Raises `TypeError: Can't instantiate abstract class Shape with abstract method area`).
   - ✅ Always instantiate concrete subclasses like `circle = Circle(5)`.

2. **Relying on Name Mangling for True Security:**
   - Python mangles `__private` to `_ClassName__private`. It prevents accidental overrides, but isn't encrypted security. Use getter/setter encapsulation rules.

3. **Complex Multiple Inheritance & Diamond Problem:**
   - Always check `ClassName.mro()` or `ClassName.__mro__` to inspect class resolution order when using multiple inheritance.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is Method Resolution Order (MRO) in Python?
**Answer:** MRO is the order in which Python searches parent classes for a method or attribute when multiple inheritance is involved. Python uses the C3 Linearization algorithm. You can inspect any class's MRO using `Class.mro()`.

### Q2: What is the difference between Abstraction and Encapsulation?
**Answer:** **Encapsulation** hides internal state data (e.g. private `__balance`) and controls how it is modified via methods. **Abstraction** hides implementation complexity entirely and enforces an architectural interface/contract via Abstract Base Classes (`ABC`).

---

## 📝 Recap Checklist
- [x] Implemented Single, Multilevel, and Multiple Inheritance.
- [x] Overrode parent methods and delegated via `super().__init__()`.
- [x] Demonstrated Polymorphism across heterogeneous object lists.
- [x] Implemented Encapsulation with private `__attributes` and Getters/Setters.
- [x] Enforced contracts using Abstract Base Classes (`ABC`) and `@abstractmethod`.
