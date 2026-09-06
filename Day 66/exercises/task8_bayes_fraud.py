"""
Task 8 — Bayes' Theorem: Fraud Detection & The Base Rate Fallacy
Calculate P(Fraud | Flagged) given low base rate and high detector accuracy.
"""

# Parameters
p_fraud = 0.01          # Prior probability: 1% of transactions are fraud
p_legit = 1 - p_fraud   # 99% are legitimate

p_flag_given_fraud = 0.95  # Sensitivity: 95% of fraud detected
p_flag_given_legit = 0.05  # False positive rate: 5% of legit flagged

# Law of Total Probability for evidence P(Flagged)
p_flagged = (p_flag_given_fraud * p_fraud) + (p_flag_given_legit * p_legit)

# Bayes' Theorem: P(Fraud | Flagged)
p_fraud_given_flag = (p_flag_given_fraud * p_fraud) / p_flagged

print("=== TASK 8: BAYES' THEOREM & BASE RATE FALLACY ===")
print(f"Prior P(Fraud):                   {p_fraud * 100:.1f}%")
print(f"Sensitivity P(Flag | Fraud):      {p_flag_given_fraud * 100:.1f}%")
print(f"False Positive P(Flag | Legit):   {p_flag_given_legit * 100:.1f}%")
print(f"Total Evidence P(Flagged):        {p_flagged * 100:.3f}%")
print("-" * 55)
print(f"P(Fraud | Flagged) = [P(Flag | Fraud) * P(Fraud)] / P(Flagged)")
print(f"                   = [{p_flag_given_fraud} * {p_fraud}] / {p_flagged:.4f}")
print(f"                   = {p_fraud_given_flag:.4f} ({p_fraud_given_flag * 100:.1f}%)")
print("-" * 55)
print("Conclusion: Even with 95% accuracy, only 16.1% of flagged transactions are")
print("actually fraudulent, because legitimate transactions vastly outnumber fraud.")
