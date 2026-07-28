# ==============================================================================
# Program    : Check if a string starts with a given letter
# Objective  : Practice and master check if a string starts with a given letter logic.
# Concept    : Using startswith() method (case-insensitive check)
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
text = input("Enter a string: ").strip()
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
target_letter = input("Enter target starting letter: ").strip()

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if not text or not target_letter:
    print("Inputs cannot be empty.")
else:
    # Perform case-insensitive check
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if text.lower().startswith(target_letter.lower()):
        print(f"[PASS] '{text}' starts with '{target_letter}'.")
    else:
        print(f"[FAIL] '{text}' does NOT start with '{target_letter}'.")
