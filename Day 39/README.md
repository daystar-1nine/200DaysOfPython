# Day 39: Payment Domain Model (Dataclasses, Enums & Protocols)

A modern, type-safe Python Payment Domain Model built with `@dataclass`, `Enum`, `typing.Protocol`, and `mypy` static type hints.

## Domain Architecture

```text
               PaymentProcessor (Protocol)
                            │
      ┌─────────────────────┼─────────────────────┬─────────────────────┐
      ↓                     ↓                     ↓                     ↓
UPIPaymentProcessor  CardPaymentProcessor  BankPaymentProcessor  WalletPaymentProcessor
```

- **Dataclasses:** `Payment` (frozen value object) and `Transaction` (stateful entity).
- **Enums:** `PaymentMethod` and `TransactionStatus`.
- **Protocols:** `PaymentProcessor` defining structural subtyping duck-typing contracts.

## Execution & Testing

```bash
# Run Payment Service demo
python Day\ 39/payment_service.py

# Run Pytest suite (15+ tests)
pytest Day\ 39/tests/test_payment.py
```
