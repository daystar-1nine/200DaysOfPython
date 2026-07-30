# ==============================================================================
# Program    : Secure Password Generator
# Objective  : Generate strong, randomized passwords using random and string modules.
# Concept    : Character Pools & Random Sampling (random.choice, string constants, random.shuffle)
# Why Used   : string module provides predefined ASCII character sets; random picks and shuffles characters.
# ==============================================================================

# What is used : import random, import string
# Why it is used: string provides character set constants; random provides selection and shuffling utilities
import random
import string

def generate_password(length=12):
    if length < 4:
        raise ValueError("Password length must be at least 4 to include all character types!")

    # What is used : Predefined character set pools from string module
    # Why it is used: Guarantees password contains uppercase, lowercase, digits, and special symbols
    uppercase = string.ascii_uppercase  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lowercase = string.ascii_lowercase  # 'abcdefghijklmnopqrstuvwxyz'
    digits = string.digits              # '0123456789'
    symbols = string.punctuation        # '!@#$%^&*...'

    # What is used : Guaranteed initial character selection using random.choice()
    # How it works : Picks 1 mandatory character from each of the 4 pools to satisfy complexity rules
    password_chars = [
        random.choice(uppercase),
        random.choice(lowercase),
        random.choice(digits),
        random.choice(symbols)
    ]

    # What is used : String concatenation (+)
    # Why it is used: Combines all character sets into single pool for remaining characters
    all_chars = uppercase + lowercase + digits + symbols

    # What is used : for loop with range(length - 4)
    # How it works : Fills remaining password slots with random characters from combined pool
    for _ in range(length - 4):
        password_chars.append(random.choice(all_chars))

    # What is used : random.shuffle(list)
    # Why it is used: Shuffles list elements in-place to randomize position of initial mandatory characters
    # How it works : Applies Fisher-Yates shuffle algorithm on password_chars list
    random.shuffle(password_chars)

    # What is used : String method "".join(list)
    # Why it is used: Converts list of individual character strings into single combined password string
    return "".join(password_chars)

def main():
    print("=== Secure Password Generator ===")
    try:
        len_input = input("Enter desired password length (default 12): ").strip()
        length = int(len_input) if len_input else 12

        # What is used : Function invocation
        pwd = generate_password(length)
        print(f"\nGenerated Secure Password: {pwd}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
