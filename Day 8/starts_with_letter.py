# ==============================================================================
# Program    : Check if a string starts with a given letter
# Objective  : Practice and master check if a string starts with a given letter logic.
# Concept    : Using startswith() method (case-insensitive check)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

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
