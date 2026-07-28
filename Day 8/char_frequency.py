# ==============================================================================
# Program    : Count the frequency of each character in a string
# Objective  : Practice and master count the frequency of each character in a string logic.
# Concept    : Dictionary-based frequency counting
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

text = input("Enter a string: ")

frequency = {}

for char in text:
    frequency[char] = frequency.get(char, 0) + 1

print("\n--- Character Frequencies ---")
for char, count in frequency.items():
    display_char = repr(char) if char in " \t\n" else char
    print(f"Character {display_char:<6} : {count} time(s)")
