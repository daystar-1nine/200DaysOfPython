# ==============================================================================
# Program    : Count vowels and consonants in a string
# Objective  : Practice and master count vowels and consonants in a string logic.
# Concept    : Iteration, membership operators, and character type checking (.isalpha())
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

text = input("Enter a string: ")

vowels_count = 0
consonants_count = 0
vowels = "aeiouAEIOU"

for char in text:
    if char.isalpha():
        if char in vowels:
            vowels_count += 1
        else:
            consonants_count += 1

print(f"Original Text: {text}")
print(f"Vowels Count    : {vowels_count}")
print(f"Consonants Count: {consonants_count}")
