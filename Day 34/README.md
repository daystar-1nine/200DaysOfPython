# Day 34: Custom Iterator Library

A modular Python library demonstrating the iteration protocol (`__iter__()`, `__next__()`, `StopIteration`).

## Provided Iterators

- **`CountdownIterator(start)`:** Iterates downwards from `start` to `1`.
- **`EvenNumberIterator(limit)`:** Generates even numbers up to `limit`.
- **`PaginationIterator(data, page_size)`:** Chunks datasets into structured page objects.
- **`TransactionIterator(transactions, min_amount)`:** Streams transaction records filtered by minimum amount.

## Execution & Testing

```bash
# Run Pytest suite
pytest Day\ 34/tests/test_iterators.py
```
