# 🧪 Day 48 — Professional Testing with Pytest

## Executive Summary

Software quality in modern engineering is guaranteed through **automated testing**. Manual testing ("clicking around in the UI or manually executing Postman calls") is slow, error-prone, and fails to catch regression bugs as codebases expand.

Today's masterclass covers **Pytest Fundamentals**, **Assertions**, **Fixtures & `conftest.py`**, **FastAPI `TestClient`**, **Dependency Overriding**, **Mocking External Services (`unittest.mock`)**, **Atomic Transaction Rollback Testing**, **Parameterized Testing (`@pytest.mark.parametrize`)**, and **Code Coverage (`pytest-cov`)**.

---

## Key Technical Concepts

### 1. Why Automated Testing Matters

```text
       CODE CHANGE
            │
            ▼
    AUTOMATED TEST SUITE (50+ Tests)
            │
      ┌─────┴─────┐
      ▼           ▼
   PASSED       FAILED (Regression Caught Instantly!)
   (Deploy)     (Fix before deployment)
```

- **Regression Prevention:** Ensures that adding a new feature or refactoring code does not break pre-existing functionality.
- **Documentation:** Automated tests serve as executable, unambiguous documentation of expected application behavior.
- **Continuous Integration (CI):** Automated test suites run on every Git commit in GitHub Actions or CI/CD pipelines.

---

### 2. Pytest Basics & Test Discovery

Pytest automatically discovers and executes test files matching:
- `test_*.py`
- `*_test.py`

And test functions named:
- `def test_*():`

#### Assertions:
In Pytest, standard Python `assert` statements are used:
```python
assert response.status_code == 200
assert "access_token" in response.json()
assert user.role == "admin"
```

---

### 3. Pytest Fixtures & `conftest.py`

A **Fixture** (`@pytest.fixture`) provides reusable setup and teardown logic for test functions.

```text
             conftest.py (Shared Fixtures)
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
  engine / db        TestClient          test_user / token
```

- **`conftest.py`:** Pytest automatically loads fixtures defined in `conftest.py` across all test files in the directory without requiring explicit imports.
- **Fixture Scopes:** `function` (default, re-run per test), `class`, `module`, or `session` (run once per test session).

---

### 4. FastAPI `TestClient` & Dependency Overrides

FastAPI provides `TestClient` (built on `httpx`), allowing tests to make simulated HTTP requests against route handlers in memory without spawning a network server listener.

#### Database Dependency Overriding:
To prevent test runs from polluting or wiping production databases, override database dependencies in `app.dependency_overrides`:

```python
app.dependency_overrides[get_db] = override_get_db
```

---

### 5. Unit Tests vs Integration Tests

| Metric | Unit Tests | Integration Tests |
| :--- | :--- | :--- |
| **Scope** | Isolates a single function or class in memory. | Tests multiple components working together (API + Service + DB). |
| **Speed** | Blazing fast (milliseconds). | Fast to moderate. |
| **Dependencies** | External DBs/APIs are mocked or stubbed. | Interacts with test databases or test engine. |
| **Example** | `test_password_is_hashed()` | `test_create_order_success(client)` |

---

### 6. Mocking External Services (`unittest.mock`)

> [!IMPORTANT]
> Never call third-party external production APIs (e.g., Stripe, PayPal, SendGrid) inside automated test suites! Network instability, rate limits, or real charges will break builds.

Use `unittest.mock.patch` or `MagicMock` to simulate third-party responses:

```python
with patch("app.services.payment_service.PaymentGateway.charge") as mock_charge:
    mock_charge.return_value = {"transaction_id": "tx_123", "status": "success"}
    response = client.post("/payments/charge", json={"amount": 100.0})
    mock_charge.assert_called_once_with(100.0, "USD")
```

---

### 7. Parameterized Testing (`@pytest.mark.parametrize`)

Instead of copying and pasting identical test functions for different inputs, parameterize tests:

```python
@pytest.mark.parametrize("invalid_email", ["no_at_sign.com", "plainaddress", "@missingname.com"])
def test_invalid_email_registration_fails(client, invalid_email):
    res = client.post("/auth/register", json={"name": "Test", "email": invalid_email, "password": "Pass123!"})
    assert res.status_code == 422
```

---

## ❓ Practice & Interview Q&A (All 20 Questions)

### Q1: Why do developers write automated tests?
**Answer:** Automated tests verify code correctness, catch regressions immediately upon code changes, document application requirements, and enable continuous integration deployments.

### Q2: What is pytest?
**Answer:** Pytest is a feature-rich, industry-standard Python testing framework that provides simple test discovery, fixture management, and readable assertion failures.

### Q3: What is an assertion?
**Answer:** An assertion (`assert condition`) tests whether a specific boolean condition evaluates to `True`. If `False`, Pytest stops execution and marks the test as failed.

### Q4: What is a pytest fixture?
**Answer:** A fixture is a function decorated with `@pytest.fixture` that prepares test state, data models, database connections, or mock clients and supplies them to test functions as arguments.

### Q5: What is `conftest.py`?
**Answer:** `conftest.py` is a special Pytest configuration file used to share fixtures across multiple test files without importing them explicitly.

### Q6: What is the difference between a unit test and an integration test?
**Answer:** A unit test verifies an isolated function or component in memory without external dependencies. An integration test verifies multiple components working together (e.g., FastAPI router + Service + Database).

### Q7: What is mocking?
**Answer:** Mocking is replacing a real object, external service, or network call with a simulated object (`MagicMock`) that records invocations and returns canned responses.

### Q8: Why should external services be mocked during testing?
**Answer:** Mocking prevents network latency, rate limit errors, API costs, and unintended external state mutations during test execution.

### Q9: What is FastAPI `TestClient`?
**Answer:** `TestClient` is a test utility wrapping Starlette/httpx that allows sending HTTP requests (`GET`, `POST`, etc.) directly to FastAPI route handlers in memory.

### Q10: How do you test a protected FastAPI endpoint?
**Answer:** Obtain a valid JWT access token, and pass it in the HTTP request headers as `headers={"Authorization": f"Bearer {token}"}`.

### Q11: How do you test admin authorization in FastAPI?
**Answer:** Execute the request using a regular user token to assert `403 Forbidden`, and execute using an admin token to assert `200 OK` or `201 Created`.

### Q12: What is dependency overriding in FastAPI?
**Answer:** `app.dependency_overrides[original_dependency] = test_dependency` swaps production dependencies (like real DB sessions) with test fixtures during test runs.

### Q13: Why should tests use a separate database?
**Answer:** Running tests on a dedicated test database prevents accidental deletion or corruption of development/production data and ensures predictable test isolation.

### Q14: What is parameterized testing?
**Answer:** Parameterized testing (`@pytest.mark.parametrize`) executes a single test function multiple times with different input values and expected outputs.

### Q15: What is code coverage?
**Answer:** Code coverage (`pytest-cov`) measures the percentage of source code lines executed during test runs.

### Q16: Does 100% code coverage mean code is 100% bug-free?
**Answer:** No. Coverage measures which lines executed, not whether assertions were thorough or edge cases were handled correctly.

### Q17: How do you test transaction rollback on failure?
**Answer:** Trigger a multi-step operation where a step fails (e.g., out-of-stock item 2 in an order), catch the exception, and assert that no partial database records were persisted.

### Q18: Why should test cases be independent?
**Answer:** Independent tests can be run in any order or in parallel without relying on side effects from previous tests.

### Q19: What is the purpose of `yield` in a Pytest fixture?
**Answer:** Code before `yield` executes as test setup; code after `yield` executes as teardown cleanup after the test completes.

### Q20: What is Test-Driven Thinking (TDD)?
**Answer:** Designing test specifications for expected behavior (e.g., 401 unauthenticated, 403 unauthorized, 200 success) before writing the actual feature code.

---

## 📋 Day 48 Recap Checklist

- [x] Installed `pytest`, `pytest-cov`, and `httpx`.
- [x] Structured production test suite in `tests/`.
- [x] Configured shared database engine and migration fixtures in `conftest.py`.
- [x] Created Payment Service with external API mocking (`unittest.mock.patch`).
- [x] Implemented Parameterized testing for email and password validation.
- [x] Executed and passed 50+ Pytest unit and integration test cases.
- [x] Generated code coverage report using `pytest-cov`.
