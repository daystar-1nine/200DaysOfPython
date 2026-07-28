# ==============================================================================
# Program    : Replace one word with another in a sentence
# Objective  : Practice and master replace one word with another in a sentence logic.
# Concept    : Using replace(old, new) method
# Why Used   : Pauses execution to capture interactive user input from standard input.
# ==============================================================================

# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
sentence = input("Enter original sentence: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
old_word = input("Enter word to replace: ")
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
new_word = input("Enter replacement word: ")

updated_sentence = sentence.replace(old_word, new_word)

print("\n--- Updated Result ---")
print("Original:", sentence)
print("Updated :", updated_sentence)
