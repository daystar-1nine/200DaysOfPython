# ==============================================================================
# Program    : Secure Password Generator
# Objective  : Generate strong, randomized passwords using random and string modules.
# Concept    : Character Sets & Random Sampling (random.choice, string constants)
# Why Used   : string module provides predefined ASCII character pools; random.choice picks characters safely.
# ==============================================================================

# What is used : import random, import string
# Why it is used: string provides character constants (letters, digits, punctuation); random picks characters
import random
import string

def generate_password(length=12):
    if length < 4:
        raise ValueError("Password length must be at least 4 to include all character types!")

    # What is used : Character pools from string module
    uppercase = string.ascii_uppercase  # 'ABC...XYZ'
    lowercase = string.ascii_lowercase  # 'abc...xyz'
    digits = string.digits              # '0123456789'
    symbols = string.punctuation        # '!@#$%^&*...'

    # Ensure password contains at least 1 character from each required category
    password_chars = [
        random.choice(uppercase),
        random.choice(lowercase),
        random.choice(digits),
        random.choice(symbols)
    ]

    # Combine all character pools for remaining length
    all_chars = uppercase + lowercase + digits + symbols

    # Fill remaining password length with random choices
    for _ in range(length - 4):
        password_chars.append(random.choice(all_chars))

    # What is used : random.shuffle(list)
    # Why it is used: Shuffles character positions in-place to prevent predictable initial positions
    # How it works : Applies Fisher-Yates shuffle algorithm on password_chars list
    random.shuffle(password_chars)

    # What is used : "".join(list)
    # Why it is used: Joins character list into a single password string
    return "".join(password_chars)

def main():
    print("=== Secure Password Generator ===")
    try:
        len_input = input("Enter desired password length (default 12): ").strip()
        length = int(len_input) if len_input else 12

        pwd = generate_password(length)
        print(f"\nGenerated Secure Password: {pwd}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
