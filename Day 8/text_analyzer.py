# ==============================================================================
# Program    : Challenge Project: Text Analyzer
# Objective  : Practice and master challenge project: text analyzer logic.
# Concept    : Counts characters, words, vowels, consonants, digits, spaces, & most frequent char
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

def analyze_text(text):
    total_chars = len(text)
    words = text.split()
    total_words = len(words)

    vowels_set = "aeiouAEIOU"
    total_vowels = sum(1 for c in text if c in vowels_set)
    total_consonants = sum(1 for c in text if c.isalpha() and c not in vowels_set)
    total_digits = sum(1 for c in text if c.isdigit())
    total_spaces = sum(1 for c in text if c.isspace())

    # Find most frequent character (excluding spaces)
    freq = {}
    for c in text:
        if not c.isspace():
            freq[c] = freq.get(c, 0) + 1

    most_frequent_char = max(freq, key=freq.get) if freq else "None"
    most_frequent_count = freq.get(most_frequent_char, 0)

    print("\n====================================")
    print("           TEXT ANALYZER            ")
    print("====================================")
    print(f"Total Characters        : {total_chars}")
    print(f"Total Words             : {total_words}")
    print(f"Total Vowels            : {total_vowels}")
    print(f"Total Consonants        : {total_consonants}")
    print(f"Total Digits            : {total_digits}")
    print(f"Total Spaces            : {total_spaces}")
    print(f"Most Frequent Character : '{most_frequent_char}' ({most_frequent_count} times)")
    print("====================================")

def main():
    paragraph = input("Enter a paragraph or text:\n")
    if not paragraph:
        print("Empty text provided!")
    else:
        analyze_text(paragraph)

if __name__ == "__main__":
    main()
