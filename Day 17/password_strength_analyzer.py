# ==============================================================================
# Program    : Password Strength Analyzer (Bonus Challenge)
# Objective  : Evaluate password against 5 security criteria and rate password strength.
# Concept    : Regex Security Metric Evaluation
# Why Used   : Checks length, uppercase, lowercase, numbers, and symbols to calculate strength score.
# ==============================================================================

import re

def analyze_password_strength(password):
    score = 0
    feedback = []

    # Criteria 1: Length >= 8
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("Increase length to at least 8 characters.")

    # Criteria 2: Has Uppercase
    if re.search(r"[A-Z]", password):
        score += 1
    else:
        feedback.append("Add at least 1 uppercase letter (A-Z).")

    # Criteria 3: Has Lowercase
    if re.search(r"[a-z]", password):
        score += 1
    else:
        feedback.append("Add at least 1 lowercase letter (a-z).")

    # Criteria 4: Has Number
    if re.search(r"\d", password):
        score += 1
    else:
        feedback.append("Add at least 1 numerical digit (0-9).")

    # Criteria 5: Has Special Symbol
    if re.search(r"[@$!%*?&#^_-]", password):
        score += 1
    else:
        feedback.append("Add at least 1 special character (@, $, !, %, *, ?, &, #).")

    # Determine strength rating based on score out of 5
    if score == 5:
        rating = "Strong Password [VALID]"
    elif score >= 3:
        rating = "Moderate Password [NEEDS IMPROVEMENT]"
    else:
        rating = "Weak Password [INVALID]"

    return score, rating, feedback

def main():
    print("=== PASSWORD STRENGTH ANALYZER ===")
    test_passwords = ["suraj123", "SurajSawant2026", "P@ssw0rd2026!", "abc"]

    for pwd in test_passwords:
        score, rating, tips = analyze_password_strength(pwd)
        print(f"\nPassword : '{pwd}'")
        print(f"Score    : {score} / 5")
        print(f"Rating   : {rating}")
        if tips:
            print("Tips     :", "; ".join(tips))

if __name__ == "__main__":
    main()
