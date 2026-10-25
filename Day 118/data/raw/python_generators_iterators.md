# Python Iterators, Iterables, and Generators

An iterable is any Python object capable of returning its members one at a time, implementing the `__iter__()` method.
An iterator is an object representing a stream of data that implements both `__iter__()` and `__next__()` protocols.
When no further items remain in the sequence, `__next__()` raises the `StopIteration` exception.

Generators provide a simple way to implement iterators using the `yield` keyword.
When a generator function calls `yield`, its state, local variables, and instruction pointer are suspended and returned to the caller.