# ==============================================================================
# Program    : Count word frequency in a sentence
# Objective  : Practice and master count word frequency in a sentence logic.
# Concept    : Building frequency counter dictionary using loop and .get()
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

sentence = "python is easy and python is powerful and python is fun"
print("Sentence:", sentence)

words = sentence.split()
word_count = {}

for word in words:
    # Increment count if word exists, otherwise default to 0 + 1
    word_count[word] = word_count.get(word, 0) + 1

print("\n--- Word Frequency ---")
for word, count in word_count.items():
    print(f"'{word}': {count} time(s)")
