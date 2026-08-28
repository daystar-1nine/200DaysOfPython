# ==============================================================================
# Program    : Banking System Executable CLI Simulation
# Objective  : Demonstrate Accounts, Transfers, Interest, and Strategy Payments.
# Concept    : Full Banking System Simulation
# Why Used   : Validates end-to-end execution of OOP banking principles.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from bank import Bank
from payments import UPIPayment, CardPayment

def main():
    print("==================================================")
    print("       DAY 31 - ADVANCED OOP BANKING SYSTEM      ")
    print("==================================================\n")

    bank = Bank("State Bank of Python")

    # 1. Create Savings & Current Accounts
    sa = bank.create_savings_account("SA-101", "Suraj Sawant", 10000.0)
    ca = bank.create_current_account("CA-202", "TechCorp Ltd", 5000.0)

    print(f"Created Savings Account #{sa.account_number} for {sa.holder_name} (Balance: Rs.{sa.balance:,.2f})")
    print(f"Created Current Account #{ca.account_number} for {ca.holder_name} (Balance: Rs.{ca.balance:,.2f})\n")

    # 2. Deposit & Withdraw
    sa.deposit(5000.0)
    print(f"After Deposit Rs.5,000 -> Savings Balance: Rs.{sa.balance:,.2f}")

    sa.withdraw(2000.0)
    print(f"After Withdraw Rs.2,000 -> Savings Balance: Rs.{sa.balance:,.2f}\n")

    # 3. Inter-Account Transfer
    print("Executing Transfer: Rs.3,000 from Savings (SA-101) to Current (CA-202)...")
    sa.transfer(ca, 3000.0)
    print(f"  New Savings Balance: Rs.{sa.balance:,.2f}")
    print(f"  New Current Balance: Rs.{ca.balance:,.2f}\n")

    # 4. Strategy Pattern Payment Processing
    print("Processing External Merchant Payments:")
    bank.process_payment("SA-101", 1500.0, UPIPayment("suraj@upi"))
    bank.process_payment("CA-202", 2500.0, CardPayment("4111222233334444", "TechCorp Ltd"))

    print(f"\nFinal Savings Balance: Rs.{sa.balance:,.2f}")
    print(f"Final Current Balance: Rs.{ca.balance:,.2f}\n")

    # 5. Apply Monthly Interest
    print("Applying Monthly Interest:")
    interests = bank.apply_monthly_interest()
    for acc_num, interest_amt in interests.items():
        print(f"  Account {acc_num} earned interest: Rs.{interest_amt:,.2f}")

    print(f"Updated Savings Balance after Interest: Rs.{sa.balance:,.2f}\n")

if __name__ == "__main__":
    main()
