# ==============================================================================
# Program    : Simple Interest Calculator
# Objective  : Compute simple interest and total payment amount.
# Why Used   : Uses formula SI = (P * R * T) / 100 to show financial calculations.
# ==============================================================================

# Step 1: Input principal, interest rate, and time in years
principal = float(input("Enter Principal amount (P): "))
rate = float(input("Enter annual Interest Rate (%): "))
time = float(input("Enter Time in years (T): "))

# Step 2: Calculate Simple Interest and Total Payable Amount
simple_interest = (principal * rate * time) / 100
total_amount = principal + simple_interest

# Step 3: Display financial breakdown
print("\n--- SIMPLE INTEREST BREAKDOWN ---")
print(f"Principal Amount  : Rs.{principal:.2f}")
print(f"Interest Rate     : {rate}% p.a.")
print(f"Time Period       : {time} years")
print(f"Interest Earned   : Rs.{simple_interest:.2f}")
print(f"Total Amount Due  : Rs.{total_amount:.2f}")
