# ==============================================================================
# Program    : Smart Collection (Main Project)
# Objective  : Build custom collection class implementing full dunder protocols.
# Concept    : Container Emulation & Operator Overloading Protocols
# Why Used   : Provides custom sequence behavior matching built-in Python types.
# ==============================================================================

class SmartCollection:
    def __init__(self, items: list | None = None):
        # What is used : Encapsulated Private List
        # Why it is used: Stores collection elements safely
        self._items = list(items) if items else []

    def add(self, item) -> None:
        self._items.append(item)

    def remove(self, item) -> None:
        if item not in self._items:
            raise ValueError(f"Item '{item}' not found in SmartCollection.")
        self._items.remove(item)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return SmartCollection(self._items[index])
        return self._items[index]

    def __setitem__(self, index: int, value) -> None:
        self._items[index] = value

    def __delitem__(self, index: int) -> None:
        del self._items[index]

    def __contains__(self, item) -> bool:
        return item in self._items

    def __iter__(self):
        return iter(self._items)

    def __str__(self) -> str:
        return f"SmartCollection({self._items})"

    def __repr__(self) -> str:
        return f"SmartCollection(items={self._items!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SmartCollection):
            return False
        return self._items == other._items

    def __add__(self, other: "SmartCollection") -> "SmartCollection":
        if not isinstance(other, SmartCollection):
            return NotImplemented
        return SmartCollection(self._items + other._items)


if __name__ == "__main__":
    print("=== SMART COLLECTION DEMO ===")
    col = SmartCollection()
    col.add("Python")
    col.add("SQL")
    col.add("Docker")

    print(f"Print collection: {col}")
    print(f"Length          : {len(col)}")
    print(f"First element   : {col[0]}")
    print(f"'Python' in col : {'Python' in col}")

    col2 = SmartCollection(["FastAPI", "Pytest"])
    combined = col + col2
    print(f"Combined        : {combined}")
