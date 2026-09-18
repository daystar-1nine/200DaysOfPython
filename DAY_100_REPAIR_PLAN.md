# DAY 100 REPAIR PLAN

## P0 — Critical Correctness/Security Issues
**Day 10, 47, 48, 49: Exposed Secrets**
* File: `Day 10/secure_login_system.py`, `Day 47/tests/test_auth.py`, `Day 48/tests/test_security.py`, `Day 49/tests/test_security.py`
* Problem: Hardcoded secrets/credentials detected in source code.
* Evidence: Static analysis regex caught JWT/Auth mock passwords and secret keys.
* Required behavior: Secrets must be loaded from environment variables (`.env`).
* Proposed fix: Strip secrets from Python files, replace with `os.getenv()`, create `.env.example`, ensure `.env` is gitignored.
* Risk: Low (refactoring to environment variables is standard and safe).
* Priority: P0

## P1 — Required Learning/Project Functionality
**Day 50: FastAPI Task Management Backend**
* File: `app/models/task.py`, `app/schemas/task.py`, `tests/test_tasks.py`
* Problem: Core logic is missing (`pass`/`TODO`).
* Evidence: Audit script flagged these files as containing placeholder logic.
* Required behavior: SQLAlchemy Task model, Pydantic validation, CRUD endpoints, and working tests.
* Proposed fix: Implement actual SQLAlchemy ORM models, Pydantic schemas, and write Pytest assertions.
* Risk: Medium (might require creating boilerplate `main.py` router bindings if not already present).
* Priority: P1

**Days 82-87: From-Scratch Algorithms**
* File: `coding_challenges/` and `experiments/` folders for Days 82-87.
* Problem: Blank or placeholder implementations for KNN, Naive Bayes, SVM concepts, and Forward Propagation/Loss.
* Evidence: Files are empty or contain `pass`.
* Required behavior: Pure NumPy/Python implementations of these mathematical concepts.
* Proposed fix: Write the math formulas explicitly and test them against deterministic toy datasets.
* Risk: Low (isolated educational files).
* Priority: P1

**Test Suite Quality**
* File: Assorted `test_*.py` files with 0-bytes or `pass`.
* Problem: Fake test completion.
* Evidence: 455 test files found, but many are empty, artificially inflating test counts.
* Required behavior: Tests must have meaningful assertions.
* Proposed fix: Delete useless 0-byte test files. Write meaningful assertions for required functional tests (e.g., Day 50).
* Risk: Low.
* Priority: P1

## P2 — Important Quality/Documentation Issue
**Empty File Cleanup (Days 1-81)**
* File: ~150 0-byte `.py` files.
* Problem: Accidental or abandoned scaffolding clutters the repository.
* Evidence: Audit script identified 150 0-byte files (excluding `__init__.py`).
* Required behavior: Only necessary files should exist.
* Proposed fix: Classify each file in `EMPTY_FILE_CLASSIFICATION.md` and carefully delete unnecessary scaffolding.
* Risk: Medium (need to avoid deleting required package markers or fixtures).
* Priority: P2

## P3 — Optional Improvement
**Documentation Alignment**
* File: Individual `README.md` files for early days.
* Problem: Claims of completion for missing exercises.
* Required behavior: READMEs must match actual codebase state.
* Proposed fix: Remove unverifiable claims.
* Risk: Low.
* Priority: P3
