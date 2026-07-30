# ==============================================================================
# Program    : Alphabet Generator
# Objective  : Stream uppercase or lowercase alphabet letters sequentially.
# Concept    : Character Streaming Generators (string module + yield)
# Why Used   : Streams letters 'A' through 'Z' lazily.
# ==============================================================================

import string

# What is used : Generator function 'def alphabet_gen(lowercase=False)'
# Why it is used: Yields alphabet letters one by one
def alphabet_gen(lowercase=False):
    # What is used : Character set from string module
    letters = string.ascii_lowercase if lowercase else string.ascii_uppercase
    for char in letters:
        # What is used : yield keyword
        yield char

def main():
    print("=== Alphabet Generator (A-Z) ===")
    alpha = alphabet_gen(lowercase=False)
    for letter in alpha:
        print(letter, end=" ")
    print()

if __name__ == "__main__":
    main()
