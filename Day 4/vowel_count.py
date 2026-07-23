# Program: Count Vowels in a Word
# Concept: String traversal, character membership check, case sensitivity handling, loop counters

word = input("Enter a word: ")

# Using for loop (traversing characters directly)
count = 0
for ch in word:
    if ch.lower() in "aeiou":
        count += 1

print("Number of vowels (using for) =", count)

# Using while loop (traversing using indices)
count = 0
j = 0
while j < len(word):
    if word[j].lower() in "aeiou":
        count += 1
    j += 1

print("Number of vowels (using while) =", count)
