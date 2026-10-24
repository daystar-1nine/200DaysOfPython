# Python Built-in Data Structures and Mutability

Python provides several core built-in data structures optimized for distinct access patterns. Lists are ordered, mutable sequences defined with square brackets, supporting dynamic resizing, amortized O(1) append and pop from the end, and in-place sorting. In contrast, tuples are ordered, immutable sequences defined with parentheses; their immutability allows them to be hashed and used as dictionary keys when their constituent elements are also immutable.

Sets are unordered collections of unique, hashable elements implemented via hash tables, providing O(1) average-time complexity for lookups, insertions, and membership tests via in keyword. Dictionaries represent associative arrays mapping unique keys to arbitrary values, preserving insertion order since Python 3.7. Under the hood, Python dictionaries use a compact array combined with a sparse hash table to minimize memory overhead while retaining O(1) key lookups.
