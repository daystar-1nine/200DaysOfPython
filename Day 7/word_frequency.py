# Program: Count word frequency in a sentence
# Concept: Building frequency counter dictionary using loop and .get()

sentence = "python is easy and python is powerful and python is fun"
print("Sentence:", sentence)

words = sentence.split()
word_count = {}

for word in words:
    # Increment count if word exists, otherwise default to 0 + 1
    word_count[word] = word_count.get(word, 0) + 1

print("
--- Word Frequency ---")
for word, count in word_count.items():
    print(f"'{word}': {count} time(s)")
