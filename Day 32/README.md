# Day 32: Dunder / Magic Methods & Smart Collection

A comprehensive guide and custom collection class implementing Python dunder protocols (`__str__`, `__repr__`, `__len__`, `__getitem__`, `__setitem__`, `__delitem__`, `__contains__`, `__iter__`, `__eq__`, and `__add__`).

## Features Supported

```python
from smart_collection import SmartCollection

collection = SmartCollection()
collection.add("Python")
collection.add("SQL")

len(collection)         # 2
collection[0]           # "Python"
"SQL" in collection     # True
print(collection)       # SmartCollection(['Python', 'SQL'])

for item in collection:
    print(item)

c1 = SmartCollection([1, 2])
c2 = SmartCollection([3, 4])
c3 = c1 + c2            # SmartCollection([1, 2, 3, 4])
```

## Running Tests

```bash
pytest Day\ 32/test_smart_collection.py
```
