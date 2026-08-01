# 🐍 Day 17/200 – Masterclass Notes: Regular Expressions (Regex)

🎯 **Goal:** Master text parsing, pattern matching, data extraction, validation, and string replacement using Python's built-in `re` module.

---

## 📌 Executive Summary & Key Takeaways

- **Regular Expressions (Regex):** A domain-specific mini-language used to match and manipulate string patterns.
- **Raw Strings (`r"..."`):** Always prefix regex patterns with `r` in Python (e.g. `r"\b\d+\b"`) to prevent Python's string parser from interpreting backslashes like `\n` or `\t` as escape characters.
- **Core Functions in `re` Module:**
  - `re.search(pattern, text)`: Finds first match anywhere in string; returns a Match object or `None`.
  - `re.match(pattern, text)`: Checks for match ONLY at the beginning of the string.
  - `re.findall(pattern, text)`: Returns a list of all matching string occurrences.
  - `re.finditer(pattern, text)`: Returns an iterator of Match objects yielding match spans and groups.
  - `re.sub(pattern, replacement, text)`: Replaces occurrences matching pattern with replacement.
  - `re.compile(pattern)`: Pre-compiles pattern into a Regex Object for high-performance reuse.

---

## 📖 Topic 1: Regex Metacharacters & Special Sequences

### 1.1 Metacharacters Quick Reference

| Symbol | Meaning | Example | Matches |
|---|---|---|---|
| `.` | Any single character (except newline) | `a.c` | "abc", "a1c" |
| `^` | Start of string anchor | `^Hello` | String starting with "Hello" |
| `$` | End of string anchor | `world$` | String ending with "world" |
| `*` | Zero or more occurrences | `ab*` | "a", "ab", "abbb" |
| `+` | One or more occurrences | `ab+` | "ab", "abbb" |
| `?` | Zero or one occurrence (Optional) | `colou?r` | "color", "colour" |
| `{n,m}` | Between `n` and `m` repetitions | `\d{3,5}` | 3 to 5 digits |
| `[]` | Character class / set | `[aeiou]` | Any single vowel |
| `|` | OR operator | `cat|dog` | "cat" or "dog" |
| `()` | Grouping & extraction | `(\d{3})-(\d{4})` | Captured groups |

### 1.2 Special Sequences (Escaped Character Classes)

| Sequence | Equivalent Set | Meaning |
|---|---|---|
| `\d` | `[0-9]` | Any decimal digit |
| `\D` | `[^0-9]` | Any non-digit character |
| `\w` | `[a-zA-Z0-9_]` | Any word character (letters, digits, underscore) |
| `\W` | `[^a-zA-Z0-9_]` | Any non-word character (punctuation, symbols) |
| `\s` | `[ \t\n\r\f\v]` | Any whitespace character |
| `\S` | `[^ \t\n\r\f\v]` | Any non-whitespace character |
| `\b` | Boundary | Word boundary (transition between `\w` and `\W`) |

---

## 📖 Topic 2: Standard Validation Patterns

### 2.1 Email Validation Pattern
```python
pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
```

### 2.2 Indian Phone Number Validation Pattern
```python
pattern = r"^[6-9]\d{9}$"
```

### 2.3 Password Security Pattern
- Minimum 8 characters, at least 1 uppercase, 1 lowercase, 1 digit, 1 special character:
```python
pattern = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
```

---

## ⚡ Master Cheat Sheet

```python
# Regex Master Cheat Sheet

import re

# 1. Pre-compiling pattern for high performance
email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

def is_valid_email(email):
    return bool(email_pattern.match(email))

# 2. Extracting Numbers
text = "Invoice #1042 total is Rs.4500 on 2026-08-01."
numbers = re.findall(r"\d+", text)  # ['1042', '4500', '2026', '08', '01']

# 3. Replacing Text
cleaned_text = re.sub(r"[^\w\s]", "", text)  # Removes punctuation
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Greedy vs Non-Greedy Matching:**
   - By default, `*` and `+` are **greedy**—they match as much text as possible.
   - Example: `<.*>` on `"<b>bold</b>"` matches `"<b>bold</b>"` completely.
   - Make operators **non-greedy** (lazy) by appending `?`: `<.*?>` matches `"<b>"` then `"</b>"`.

2. **Forgetting Raw Strings (`r"..."`):**
   - ❌ `re.findall("\bword\b", text)` (Python interprets `\b` as ASCII backspace `\x08`).
   - ✅ `re.findall(r"\bword\b", text)` (Correctly interprets `\b` as word boundary).

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between `re.search()` and `re.match()`?
**Answer:** `re.match()` checks for a pattern match starting strictly at index 0 (the beginning of the string). `re.search()` scans through the entire string and returns the first match found anywhere in the text.

### Q2: What is a Lookahead Assertion in Regex?
**Answer:** A lookahead assertion (e.g. positive lookahead `(?=...)`) checks if a specific pattern follows the current position without consuming characters in the match span. It is commonly used for multi-rule password validations.

---

## 📝 Recap Checklist
- [x] Used raw strings (`r"..."`) for all regex pattern definitions.
- [x] Differentiated between `re.match()`, `re.search()`, `re.findall()`, and `re.sub()`.
- [x] Used character classes (`\d`, `\w`, `\s`, `\b`) and quantifiers (`+`, `*`, `?`, `{n,m}`).
- [x] Validated emails, phone numbers, and strong passwords.
- [x] Extracted URLs, dates, hashtags, and numbers from unstructured text.
- [x] Built a Resume Information Extractor and Password Strength Analyzer.
