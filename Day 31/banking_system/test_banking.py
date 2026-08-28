# ==============================================================================
# Test Suite : Day 31 Banking System Pytest / Unittest Suite
# Objective  : Test SavingsAccount, CurrentAccount, Transfers, and Strategy Payments.
# Concept    : Unit Testing Advanced OOP Principles
# Why Used   : Asserts business invariants and abstract method enforcement.
# ==============================================================================

import os
import sys
import unittest
import pytest

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from account import SavingsAccount, CurrentAccount, Account
from bank import Bank
from payments import UPIPayment, CardPayment

def test_savings_account_min_balance():
    sa = SavingsAccount("SA-TEST", "Tester", initial_balance=1000.0, min_balance=500.0)
    with pytest.raises(ValueError, match="Must maintain minimum balance"):
        sa.withdraw(600.0)

def test_current_account_overdraft():
    ca = CurrentAccount("CA-TEST", "Tester", initial_balance=1000.0, overdraft_limit=2000.0)
    assert ca.withdraw(2500.0) is True
    assert ca.balance == -1500.0

def test_abstract_account_instantiation_raises_error():
    with pytest.raises(TypeError):
        Account("ABS-1", "Abstract", 100.0)

class TestBankingSystemRunner(unittest.TestCase):
    def test_bank_operations(self):
        bank = Bank("Test Bank")
        sa = bank.create_savings_account("S1", "Alice", 2000.0)
        ca = bank.create_current_account("C1", "Bob", 1000.0)

        self.assertTrue(sa.transfer(ca, 500.0))
        self.assertEqual(sa.balance, 1500.0)
        self.assertEqual(ca.balance, 1500.0)

        self.assertTrue(bank.process_payment("S1", 200.0, UPIPayment("alice@upi")))
        self.assertEqual(sa.balance, 1300.0)

if __name__ == "__main__":
    unittest.main()
