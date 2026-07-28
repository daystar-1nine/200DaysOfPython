# ==============================================================================
# Program    : Count words in a sentence
# Objective  : Practice and master count words in a sentence logic.
# Concept    : Using split() method to tokenize sentence by whitespace
# Why Used   : Stores ordered, mutable collections of items allowing dynamic modification. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

sentence = input("Enter a sentence: ").strip()

# Handle empty string edge case
if not sentence:
    total_words = 0
else:
    words_list = sentence.split()
    total_words = len(words_list)

print(f"Total Words: {total_words}")
