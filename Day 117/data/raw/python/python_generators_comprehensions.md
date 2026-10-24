# Python Comprehensions, Generators, and Lazy Evaluation

List comprehensions provide a concise syntax for mapping and filtering iterables, constructing a full in-memory list upon execution: [x*2 for x in items if x > 0]. While highly expressive and fast due to optimized C-level bytecode loops, list comprehensions can consume excessive RAM when dealing with large or infinite sequences.

Generator expressions solve this memory bottleneck by using parentheses instead of square brackets: (x*2 for x in items). Generators produce values lazily on-demand using the iterator protocol (__iter__ and __next__). Generator functions use the yield keyword to pause execution and yield intermediate values, maintaining internal state across invocations. The itertools module further provides memory-efficient combinatoric iterators such as chain, islice, and cycle.
