"""
===============================================================================
DAY 51 — CODING CHALLENGE 5: CUSTOMER QUEUE USING COLLECTIONS.DEQUE
===============================================================================
This module implements a customer service queue management system using
collections.deque for O(1) double-ended queue operations.
===============================================================================
"""

from collections import deque
from typing import List


class CustomerQueue:
    """Customer queue abstraction powered by collections.deque."""

    def __init__(self) -> None:
        # What is used: collections.deque initialization.
        # Why it is used: Implements O(1) queue additions and deletions.
        # How it works: Stores customer name strings in a deque double-ended list.
        self._queue: deque[str] = deque()

    def add_customer(self, name: str) -> None:
        """Enqueue customer to the back of the queue."""
        self._queue.append(name)

    def serve_customer(self) -> str:
        """Dequeue customer from the front of the queue."""
        if not self._queue:
            raise IndexError("Queue is empty.")
        return self._queue.popleft()

    def show_queue(self) -> List[str]:
        """Return snapshot list of customers currently in queue."""
        return list(self._queue)

    @property
    def count(self) -> int:
        """Return total pending queue length."""
        return len(self._queue)


if __name__ == "__main__":
    q = CustomerQueue()
    q.add_customer("Customer A")
    q.add_customer("Customer B")
    q.add_customer("Customer C")

    print("Current Queue:", q.show_queue())
    assert q.count == 3

    served1 = q.serve_customer()
    print("Served Customer:", served1)
    assert served1 == "Customer A"
    assert q.show_queue() == ["Customer B", "Customer C"]
    print("[OK] Challenge 5 Passed!")
