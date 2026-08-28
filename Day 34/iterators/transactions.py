# ==============================================================================
# Program    : Transaction Record Iterator (TransactionIterator)
# Objective  : Stream financial transaction records filtering by optional min_amount.
# Concept    : Filtered Records Iterator
# Why Used   : Provides clean single-pass stream processing over financial records.
# ==============================================================================

class TransactionIterator:
    def __init__(self, transactions: list[dict], min_amount: float = 0.0):
        if min_amount < 0:
            raise ValueError("Minimum amount cannot be negative.")
        self.transactions = transactions
        self.min_amount = min_amount
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        while self._index < len(self.transactions):
            tx = self.transactions[self._index]
            self._index += 1
            if tx.get("amount", 0.0) >= self.min_amount:
                return tx
        raise StopIteration


if __name__ == "__main__":
    print("=== TRANSACTION ITERATOR DEMO ===")
    records = [
        {"id": 1, "amount": 250.0, "category": "Food"},
        {"id": 2, "amount": 50.0, "category": "Snacks"},
        {"id": 3, "amount": 1200.0, "category": "Shopping"}
    ]
    for tx in TransactionIterator(records, min_amount=100.0):
        print(f"Transaction #{tx['id']}: Rs.{tx['amount']} -> {tx['category']}")
