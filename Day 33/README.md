# Day 33: Function Monitoring System (Decorators)

A modular Python function monitoring system built with reusable decorators (`@logger`, `@timer`, `@retry`, `@requires_auth`).

## Architecture & Decorators

- **`@logger`:** Logs function entry, positional/keyword arguments, and return values.
- **`@timer`:** Measures high-resolution execution time using `time.perf_counter()`.
- **`@retry(max_attempts=3, delay=0.1)`:** Parameterized decorator factory providing resilient retries.
- **`@requires_auth(role="admin")`:** Enforces access control permissions before function execution.

## Execution & Testing

```bash
# Run interactive demo
python Day\ 33/examples/monitor_demo.py

# Run complete Pytest test suite (12 tests)
pytest Day\ 33/tests/test_decorators.py
```
