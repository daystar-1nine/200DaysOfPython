# ==============================================================================
# Program    : Replace one word with another in a sentence
# Objective  : Practice and master replace one word with another in a sentence logic.
# Concept    : Using replace(old, new) method
# Why Used   : Pauses execution to capture interactive user input from standard input.
# ==============================================================================

sentence = input("Enter original sentence: ")
old_word = input("Enter word to replace: ")
new_word = input("Enter replacement word: ")

updated_sentence = sentence.replace(old_word, new_word)

print("\n--- Updated Result ---")
print("Original:", sentence)
print("Updated :", updated_sentence)
