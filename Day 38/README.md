# Day 38: Functional Data Transformation Pipeline

A pure functional Python data pipeline leveraging `lambda`, `map()`, `filter()`, `sorted()`, `functools.reduce()`, `any()`, and `all()`.

## Functional Architecture

```text
Raw Transactions
       ↓
Validator (all, filter)
       ↓
Filter (filter, lambda)
       ↓
Transformer (map, lambda)
       ↓
Sorter (sorted, key=lambda)
       ↓
Aggregator (reduce, any, all)
       ↓
Summary Report
```

## Execution & Testing

```bash
# Run functional pipeline demo
python Day\ 38/main_pipeline.py

# Run Pytest suite
pytest Day\ 38/tests/test_pipeline.py
```
