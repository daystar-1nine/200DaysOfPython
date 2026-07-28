# ==============================================================================
# Program    : Find the longest word in a sentence
# Objective  : Practice and master find the longest word in a sentence logic.
# Concept    : Tokenization using split() and max() with key=len
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

sentence = input("Enter a sentence: ").strip()

words = sentence.split()

if not words:
    print("No words entered.")
else:
    # Method 1: Using max() function with key=len
    longest = max(words, key=len)
    
    # Method 2: Manual iteration loop
    longest_manual = ""
    for w in words:
        if len(w) > len(longest_manual):
            longest_manual = w

    print(f"Longest Word: '{longest}' (Length: {len(longest)})")
