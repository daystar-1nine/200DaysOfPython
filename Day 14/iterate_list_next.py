# ==============================================================================
# Program    : Iterate Over a List Using iter() and next()
# Objective  : Demonstrate manual iteration protocol over a Python list.
# Concept    : Iterator Protocol (iter(), next(), StopIteration)
# Why Used   : Teaches how Python's for-loop works under the hood by converting iterables to iterators.
# ==============================================================================

# What is used : Python list iterable
# Why it is used: Serves as baseline sequence collection
fruits = ["Apple", "Banana", "Mango", "Orange"]
print("Original List:", fruits)

# What is used : Built-in iter() function
# Why it is used: Converts iterable list into an iterator object
# How it works : Calls fruits.__iter__() to obtain iterator state machine
fruit_iterator = iter(fruits)
print("\n--- Manual Iteration via next() ---")

# What is used : try-while-True loop capturing StopIteration
# Why it is used: Emulates internal for-loop mechanism
while True:
    try:
        # What is used : Built-in next() function
        # Why it is used: Retrieves next element from iterator stream
        # How it works : Calls fruit_iterator.__next__()
        item = next(fruit_iterator)
        print(f"Retrieved Item: {item}")
    except StopIteration:
        # What is used : StopIteration exception handling
        # Why it is used: Signals that iterator has no remaining elements
        print("\n[StopIteration] Iterator stream exhausted cleanly!")
        break
