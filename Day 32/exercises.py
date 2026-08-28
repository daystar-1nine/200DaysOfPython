# ==============================================================================
# Program    : Day 32 Dunder Methods Practice Exercises
# Objective  : Implement Book, ShoppingCart, Money, and Vector using dunder methods.
# Concept    : Dunder / Magic Methods Practice
# Why Used   : Demonstrates operator overloading and container emulation.
# ==============================================================================

# --- Exercise 1: Book Class ---
class Book:
    def __init__(self, title: str, author: str, pages: int):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self) -> str:
        return f"'{self.title}' by {self.author}"

    def __repr__(self) -> str:
        return f"Book(title={self.title!r}, author={self.author!r}, pages={self.pages})"


# --- Exercise 2: ShoppingCart Class ---
class ShoppingCart:
    def __init__(self, items: list[str] | None = None):
        self.items = list(items) if items else []

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]

    def __contains__(self, item: str) -> bool:
        return item in self.items


# --- Exercise 3: Money Class ---
class Money:
    def __init__(self, amount: float):
        self.amount = float(amount)

    def __add__(self, other: "Money") -> "Money":
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount + other.amount)

    def __sub__(self, other: "Money") -> "Money":
        if not isinstance(other, Money):
            return NotImplemented
        return Money(self.amount - other.amount)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return False
        return self.amount == other.amount

    def __lt__(self, other: "Money") -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount < other.amount

    def __repr__(self) -> str:
        return f"Money(amount={self.amount:.2f})"


# --- Exercise 4: Vector Class ---
class Vector:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other: "Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector") -> "Vector":
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x - other.x, self.y - other.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return False
        return self.x == other.x and self.y == other.y

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"


if __name__ == "__main__":
    print("=== DAY 32 DUNDER METHOD EXERCISES ===")

    # Ex 1
    b = Book("Clean Code", "Robert C. Martin", 464)
    print(f"Book str : {b}")
    print(f"Book repr: {repr(b)}")

    # Ex 2
    cart = ShoppingCart(["Laptop", "Mouse", "Keyboard"])
    print(f"Cart Length: {len(cart)}")
    print(f"First Item : {cart[0]}")
    print(f"'Mouse' in cart: {'Mouse' in cart}")

    # Ex 3
    m1 = Money(500)
    m2 = Money(300)
    print(f"m1 + m2: {m1 + m2}")
    print(f"m1 - m2: {m1 - m2}")
    print(f"m1 == m2: {m1 == m2}")
    print(f"m2 < m1 : {m2 < m1}")

    # Ex 4
    v1 = Vector(2, 3)
    v2 = Vector(5, 7)
    print(f"v1 + v2: {v1 + v2}")
    print(f"v2 - v1: {v2 - v1}")
    print(f"v1 == Vector(2, 3): {v1 == Vector(2, 3)}")
