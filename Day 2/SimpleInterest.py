# Program: Calculate Simple Interest
# Concept: Financial Math Formula, Multi-variable Input, and Floating-point Arithmetic

# Step 1: Input Principal, Rate of Interest, and Time from user
principal = float(input("Enter Principal Amount: "))
rate = float(input("Enter Rate of Interest (% per annum): "))
time = float(input("Enter Time in Years: "))

# Step 2: Calculate Simple Interest using formula: SI = (P * R * T) / 100
simple_interest = (principal * rate * time) / 100

# Step 3: Print calculated Simple Interest
print("Simple Interest =", round(simple_interest, 2))