# ==============================================================================
# Program    : ATM Withdrawal Eligibility Checker
# Objective  : Validate ATM PIN and account balance before processing withdrawal.
# Why Used   : Demonstrates nested conditional statements (if-else) to validate security 
#              and financial constraints sequentially.
# ==============================================================================

# Defined static balance and correct PIN for demonstration
CORRECT_PIN = 1234
balance = 10000.0

print("=== ATM WITHDRAWAL SYSTEM ===")

# Step 1: Accept PIN input from user
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
user_pin = int(input("Enter your 4-digit PIN: "))

# Step 2: Check PIN correctness (First level condition)
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if user_pin == CORRECT_PIN:
    print("PIN Verified Successfully!")
    
    # Step 3: Input withdrawal amount
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    withdrawal_amount = float(input("Enter withdrawal amount: Rs."))
    
    # Step 4: Check if account has sufficient balance (Nested condition)
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if withdrawal_amount <= balance:
        balance -= withdrawal_amount
        print("\nTransaction Successful!")
        print(f"Amount Withdrawn : Rs.{withdrawal_amount:.2f}")
        print(f"Remaining Balance: Rs.{balance:.2f}")
    else:
        print("\nTransaction Failed: Insufficient Balance!")
        print(f"Available Balance: Rs.{balance:.2f}")
else:
    print("\nTransaction Failed: Incorrect PIN!")
