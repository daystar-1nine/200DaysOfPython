# Program: Count words in a sentence
# Concept: Using split() method to tokenize sentence by whitespace

sentence = input("Enter a sentence: ").strip()

# Handle empty string edge case
if not sentence:
    total_words = 0
else:
    words_list = sentence.split()
    total_words = len(words_list)

print(f"Total Words: {total_words}")
