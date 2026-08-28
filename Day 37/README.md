# Day 37: Advanced Exception Handling & Application Error System

A production-style application error taxonomy and exception chaining system built with custom Python exceptions.

## Exception Taxonomy

```text
ApplicationError
│
├── ValidationError
│   ├── BoundsError
│   └── FormatError
│
├── DatabaseError
│   └── UniqueConstraintError
│
├── NotFoundError
│
├── ExternalServiceError
│   └── TimeoutError
│
└── AuthenticationError
```

## Execution & Testing

```bash
# Run service demo with chained exceptions
python Day\ 37/app_service.py

# Run Pytest suite (16 tests covering all exception classes and chaining)
pytest Day\ 37/tests/test_exceptions.py
```
