# Program: Check if a string starts with a given letter
# Concept: Using startswith() method (case-insensitive check)

text = input("Enter a string: ").strip()
target_letter = input("Enter target starting letter: ").strip()

if not text or not target_letter:
    print("Inputs cannot be empty.")
else:
    # Perform case-insensitive check
    if text.lower().startswith(target_letter.lower()):
        print(f"[PASS] '{text}' starts with '{target_letter}'.")
    else:
        print(f"[FAIL] '{text}' does NOT start with '{target_letter}'.")
