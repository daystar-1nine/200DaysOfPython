# Python List Comprehensions and Generator Expressions

Python list comprehension provides a concise and readable syntax for creating new lists from existing iterables.
The canonical syntax is `[expression for item in iterable if condition]`.
List comprehensions evaluate eagerly and construct the full list in memory immediately.

In contrast, generator expressions use parentheses `(expression for item in iterable if condition)` and produce iterators that yield items lazily on demand.
Generator expressions have an O(1) memory footprint regardless of stream size, making them ideal for unbounded or large datasets.

Best practices recommend avoiding nested list comprehensions deeper than two levels to preserve code readability.
For simple map-and-filter workflows, list comprehensions execute significantly faster than manual `for` loops appending to an empty list because the bytecode executes at C-speed in CPython.