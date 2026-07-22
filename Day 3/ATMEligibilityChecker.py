# Program: ATM Withdrawal Eligibility Checker
# Concept: Nested conditional statements, comparison operators, and logical 'and' operator.

# Step 1: Initialize account dummy credentials and database
correct_pin = "1234"
account_balance = 50000.00
card_active = True

print("========== Welcome to Python ATM ==========")

# Step 2: Input card PIN
user_pin = input("Enter your 4-digit PIN: ")

# Step 3: Check if PIN is correct
if user_pin == correct_pin:
    # Step 4: Check if card is active (nested condition)
    if card_active:
        # Step 5: Input withdrawal amount
        withdrawal_amount = float(input(f"Enter withdrawal amount (Current Balance: Rs.{account_balance}): "))
        
        # Step 6: Validate withdrawal limit and balance eligibility
        if withdrawal_amount <= 0:
            print("Error: Invalid withdrawal amount!")
        elif withdrawal_amount <= account_balance:
            account_balance -= withdrawal_amount
            print("\nTransaction Successful!")
            print(f"Please collect your cash.")
            print(f"Remaining Balance: Rs.{account_balance}")
        else:
            print("\nError: Insufficient balance!")
    else:
        print("\nError: Your card is blocked. Please contact your bank.")
else:
    print("\nError: Incorrect PIN! Access Denied.")

print("===========================================")
