# Program: Count vowels and consonants in a string
# Concept: Iteration, membership operators, and character type checking (.isalpha())

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
