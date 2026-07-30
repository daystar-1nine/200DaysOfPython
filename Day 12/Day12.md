# 🐍 Day 12/200 – Masterclass Notes: Object-Oriented Programming (OOP) – Part 1

🎯 **Goal:** Master the core principles of Object-Oriented Programming (OOP) in Python including Classes, Objects, Constructors (`__init__`), Attributes, Instance Methods, and the `self` keyword to write scalable, reusable, and structured applications.

---

## 📌 Executive Summary & Key Takeaways

- **Object-Oriented Programming (OOP):** A paradigm that structures programs around **objects** (data structures combining state/attributes and behavior/methods) rather than functions and logic alone.
- **Class vs Object:**
  - **Class:** A user-defined template/blueprint (e.g. `Car` design blueprint).
  - **Object (Instance):** A concrete instance built from the class template residing in memory (e.g. a specific red Tesla Model 3).
- **The Constructor (`__init__`):** A special double-underscore ("dunder") magic method that executes automatically when a new object instance is instantiated.
- **The `self` Parameter:** A reference to the specific current instance calling the method. Python passes `self` automatically as the first parameter of any instance method.

---

## 📖 Topic 1: Procedural vs Object-Oriented Programming

| Feature | Procedural Programming | Object-Oriented Programming (OOP) |
|---|---|---|
| **Focus** | Functions, operations, and sequence of execution | Objects combining data (state) and behavior |
| **Data Security** | Data floats freely across functions | Data is encapsulated within class instances |
| **Reusability** | Code reuse via function calls | High reuse via classes, objects, and inheritance |
| **Scalability** | Harder to maintain as project size grows | Highly modular and maintainable for complex systems |

---

## 📖 Topic 2: Classes, Objects, and the `__init__` Constructor

### 2.1 Anatomy of a Python Class

```python
class Student:
    """Class representing a student entity."""

    # 1. Constructor Method (__init__)
    def __init__(self, name, roll_no, marks):
        # Instance Attributes (bound to 'self')
        self.name = name
        self.roll_no = roll_no
        self.marks = marks

    # 2. Instance Method
    def display_info(self):
        print(f"Student: {self.name} | Roll No: {self.roll_no} | Marks: {self.marks}")

# Object Instantiation
student1 = Student("Suraj", 101, 92)
student2 = Student("Rahul", 102, 85)

# Method Invocation
student1.display_info()
student2.display_info()
```

---

## 📖 Topic 3: The `self` Keyword Explained

When you invoke a method on an object:
```python
student1.display_info()
```
Python automatically translates this invocation behind the scenes into:
```python
Student.display_info(student1)
```
The object `student1` is explicitly passed into the first parameter `self`. That's why every instance method in Python must list `self` as its first parameter!

---

## ⚡ Master Cheat Sheet

```python
# OOP Part 1 Cheat Sheet

class BankAccount:
    def __init__(self, owner, initial_balance=0.0):
        self.owner = owner          # Attribute
        self.balance = initial_balance  # Attribute

    def deposit(self, amount):      # Instance Method
        if amount > 0:
            self.balance += amount
            return True
        return False

# Creating instances
acc = BankAccount("Suraj", 5000)
acc.deposit(1500)
print(f"Owner: {acc.owner}, Balance: Rs.{acc.balance}")
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Forgetting `self` in Method Definitions:**
   - ❌ `def greet(): print("Hello")` inside a class (Raises `TypeError: greet() takes 0 positional arguments but 1 was given`).
   - ✅ `def greet(self): print(f"Hello from {self.name}")`.

2. **Forgetting `self.` when Accessing Instance Attributes:**
   - ❌ `print(name)` inside a class method (Raises `NameError`).
   - ✅ `print(self.name)` (Correctly references instance variable).

3. **Modifying Attributes Directly without Validation:**
   - Prefer writing validation methods (like `deposit(amount)` or `update_salary(new_salary)`) to ensure data integrity rather than mutating raw attributes blindly.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between an Attribute and a Method in Python OOP?
**Answer:** An **Attribute** is a variable that stores state/data associated with an object (e.g. `self.speed = 100`). A **Method** is a function defined inside a class that defines behaviors or operations that an object can perform (e.g. `def accelerate(self):`).

### Q2: What happens if you do not define an `__init__` constructor in a class?
**Answer:** Python provides a default parameterless constructor inherited from `object`. You can still instantiate the class, but instance attributes won't be initialized upon creation unless assigned afterwards manually.

---

## 📝 Recap Checklist
- [x] Defined classes as blueprints and instantiated unique objects.
- [x] Implemented constructor methods (`__init__`) to bind attributes.
- [x] Understood the mechanics of the `self` keyword.
- [x] Built domain classes (`Student`, `Car`, `Employee`, `Book`, `Mobile`, `BankAccount`).
- [x] Encapsulated state and behaviors into structured instance methods.
