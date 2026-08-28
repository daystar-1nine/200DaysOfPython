# Day 36: Context Managers & Resource Manager

A comprehensive Python resource management system built with class-based and generator-based context managers (`__enter__()`, `__exit__()`, and `@contextmanager`).

## Resource Managers Included

- **`DatabaseManager` & `transaction`:** SQLite connection management with automatic `COMMIT` on normal exit or `ROLLBACK` on unhandled exception.
- **`TimerManager` & `execution_timer`:** High-resolution execution time profilers for code block statements.
- **`TemporaryFileManager` & `temp_file`:** Safe temporary file creation with guaranteed automatic deletion upon exiting the `with` block.

## Execution & Testing

```bash
# Run interactive demo
python Day\ 36/resource_manager/context_demo.py

# Run Pytest suite
pytest Day\ 36/tests/test_resource_manager.py
```
