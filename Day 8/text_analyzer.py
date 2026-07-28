# ==============================================================================
# Program    : Challenge Project: Text Analyzer
# Objective  : Practice and master challenge project: text analyzer logic.
# Concept    : Counts characters, words, vowels, consonants, digits, spaces, & most frequent char
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================


# What is used : Function definition 'def analyze_text'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
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

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for c in text:
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
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


# What is used : Function definition 'def main'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def main():
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
    paragraph = input("Enter a paragraph or text:\n")
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if not paragraph:
        print("Empty text provided!")
    else:
        analyze_text(paragraph)

# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if __name__ == "__main__":
    main()
