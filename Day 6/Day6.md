# 🐍 Day 6/200 – Lists in Python

🎯 **Goal:** Learn how to store multiple values in a single variable and perform common list operations such as adding, removing, updating, slicing, and iterating.

---

## 1. What is a List?
A **list** is a built-in Python data structure used to store a collection of items in a single variable. Lists are **ordered**, **mutable** (changeable), and allow **duplicate values**.

```python
# Creating a list of strings
fruits = ["Apple", "Banana", "Mango", "Orange"]
print(fruits)  # Output: ['Apple', 'Banana', 'Mango', 'Orange']

# A list can contain different data types
mixed_list = [10, "Hello", 3.14, True]
print(mixed_list)
```

---

## 2. Accessing List Elements
List elements are indexed starting from `0`. You can also use negative indexing (`-1` refers to the last element).

```python
fruits = ["Apple", "Banana", "Mango"]

# Positive Indexing
print(fruits[0])   # Output: Apple
print(fruits[1])   # Output: Banana

# Negative Indexing
print(fruits[-1])  # Output: Mango (last element)
print(fruits[-2])  # Output: Banana (second last element)
```

---

## 3. Updating a List
Since lists are mutable, you can change the value of a specific element using its index.

```python
fruits = ["Apple", "Banana", "Mango"]

# Changing "Banana" to "Grapes"
fruits[1] = "Grapes"
print(fruits)  # Output: ['Apple', 'Grapes', 'Mango']
```

---

## 4. Adding Elements
You can add elements to a list using methods like `append()` and `insert()`.

```python
fruits = ["Apple", "Banana", "Mango"]

# append() adds an element to the END of the list
fruits.append("Pineapple")
print(fruits)  # Output: ['Apple', 'Banana', 'Mango', 'Pineapple']

# insert(index, item) inserts an element at a SPECIFIC position
fruits.insert(1, "Kiwi")
print(fruits)  # Output: ['Apple', 'Kiwi', 'Banana', 'Mango', 'Pineapple']
```

---

## 5. Removing Elements
Python provides several ways to remove items from a list:

```python
fruits = ["Apple", "Kiwi", "Banana", "Mango", "Pineapple"]

# remove(item) removes the FIRST occurrence of a specific item
fruits.remove("Apple")
print(fruits)  # Output: ['Kiwi', 'Banana', 'Mango', 'Pineapple']

# pop(index) removes and returns element at given index (defaults to last item)
popped_item = fruits.pop()
print(popped_item)  # Output: Pineapple
print(fruits)       # Output: ['Kiwi', 'Banana', 'Mango']

# del removes item at specified index or deletes the entire list
del fruits[0]
print(fruits)  # Output: ['Banana', 'Mango']
```

---

## 6. List Slicing
Slicing allows you to access a range of elements in a list using `list[start:stop:step]`.

```python
numbers = [1, 2, 3, 4, 5, 6, 7]

print(numbers[1:5])   # Output: [2, 3, 4, 5] (Index 1 to 4)
print(numbers[:4])    # Output: [1, 2, 3, 4] (Start to index 3)
print(numbers[3:])    # Output: [4, 5, 6, 7] (Index 3 to end)
print(numbers[::-1])  # Output: [7, 6, 5, 4, 3, 2, 1] (Reversed list)
```

---

## 7. Loop Through a List
You can iterate over items in a list using a `for` loop.

```python
fruits = ["Apple", "Banana", "Mango"]

# Iterating over items directly
for fruit in fruits:
    print(fruit)

# Iterating using index
for i in range(len(fruits)):
    print(f"Index {i}: {fruits[i]}")
```

---

## 8. Useful List Methods

| Method | Description | Example |
|---|---|---|
| `append(x)` | Adds `x` to the end of the list | `numbers.append(10)` |
| `insert(i, x)` | Inserts `x` at index `i` | `numbers.insert(0, 5)` |
| `remove(x)` | Removes first item equal to `x` | `numbers.remove(5)` |
| `pop([i])` | Removes and returns item at index `i` (default last) | `item = numbers.pop()` |
| `sort()` | Sorts list in ascending order in-place | `numbers.sort()` |
| `reverse()` | Reverses the elements in-place | `numbers.reverse()` |
| `count(x)` | Returns number of times `x` appears | `numbers.count(2)` |
| `index(x)` | Returns index of first element equal to `x` | `numbers.index(3)` |
| `clear()` | Removes all items from the list | `numbers.clear()` |
| `len(list)` | Returns total number of items in list | `len(numbers)` |

### Example:
```python
numbers = [5, 3, 8, 1, 3]

numbers.sort()
print(numbers)  # Output: [1, 3, 3, 5, 8]

numbers.reverse()
print(numbers)  # Output: [8, 5, 3, 3, 1]

print(numbers.count(3))  # Output: 2
```

---

## 9. List Comprehension (Introduction)
List comprehension offers a shorter syntax when you want to create a new list based on the values of an existing sequence.

```python
# Traditional approach
squares = []
for x in range(10):
    squares.append(x * x)

# List Comprehension approach
squares = [x * x for x in range(10)]
print(squares)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

---

## 💡 Summary of Key Concepts
- Lists store multiple items in a single variable.
- Lists are ordered, indexed, mutable, and allow duplicates.
- Use `append()` to add to the end, `insert()` to add at index.
- Use `remove()`, `pop()`, or `del` to remove items.
- Use slicing `[start:stop:step]` to extract sublists.
