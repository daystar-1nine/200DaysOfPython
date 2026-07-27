# Program: Count the frequency of each character in a string
# Concept: Dictionary-based frequency counting

text = input("Enter a string: ")

frequency = {}

for char in text:
    frequency[char] = frequency.get(char, 0) + 1

print("\n--- Character Frequencies ---")
for char, count in frequency.items():
    display_char = repr(char) if char in " \t\n" else char
    print(f"Character {display_char:<6} : {count} time(s)")
