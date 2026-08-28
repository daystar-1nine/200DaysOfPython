# 🐍 Day 24/200 – Masterclass Notes: CLI Applications & Configuration

🎯 **Goal:** Transform Python scripts into production-ready Command-Line Interface (CLI) applications using `argparse`, subcommands, positional/optional arguments, flags, environment variables (`os.getenv()`, `python-dotenv`), `.env.example`, and JSON configurations.

---

## 📌 Executive Summary & Key Takeaways

- **What is a CLI?** A Command Line Interface enables developers and automation scripts to interact with software by passing arguments and flags directly in the shell terminal (e.g. `git commit -m "msg"`, `pip install pkg`).
- **`sys.argv` vs `argparse`:**
  - `sys.argv`: Primitive list of raw string arguments passed via command line (`sys.argv[0]` is script name). Lacks type conversion, flag parsing, subcommands, or automatic `--help` documentation.
  - `argparse`: Built-in standard library module providing robust argument parsing, automatic `--help` generation, type validation, default values, subcommands, and flag handling.
- **Subparsers / Subcommands:** Allows grouping related functionality under sub-commands (e.g. `expense add`, `expense list`, `expense delete`).
- **Configuration Management:**
  - **Secrets & Credentials:** Store in `.env` files (never committed to Git) and load via `os.getenv()` or `python-dotenv`. Provide `.env.example` in repository.
  - **Application Settings:** Non-sensitive settings stored in `config.json` or YAML.

---

## 📖 Topic 1: Argument Parsing with `argparse`

### 1.1 Positional, Optional & Flag Arguments

```python
import argparse

parser = argparse.ArgumentParser(description="Professional CLI Tool Example")

# Positional Argument (Required)
parser.add_argument("name", type=str, help="User's full name")

# Optional Argument with default value
parser.add_argument("--age", type=int, default=18, help="User's age (default: 18)")

# Boolean Flag (Default: False, set to True if flag is passed)
parser.add_argument("--formal", action="store_true", help="Enable formal greeting mode")

args = parser.parse_args()

if args.formal:
    print(f"Good day, {args.name}. Age: {args.age}")
else:
    print(f"Hello {args.name}! Age: {args.age}")
```

---

## 📖 Topic 2: Subcommands / Subparser Architecture

```python
import argparse

parser = argparse.ArgumentParser(description="Multi-Command CLI Application")
subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

# 1. 'add' Subcommand
add_parser = subparsers.add_parser("add", help="Add a new item")
add_parser.add_argument("--item", type=str, required=True, help="Item description")

# 2. 'list' Subcommand
list_parser = subparsers.add_parser("list", help="List all items")

args = parser.parse_args()

if args.command == "add":
    print(f"Adding item: {args.item}")
elif args.command == "list":
    print("Listing items...")
```

---

## 📖 Topic 3: Configuration & Environment Variables (`.env`)

```python
import os
import json
from dotenv import load_dotenv

# Load .env variables into environment
load_dotenv()

# Access environment variable safely
api_key = os.getenv("API_KEY", "default_secret")
app_name = os.getenv("APP_NAME", "CLI App")

# Load JSON Config
def load_config(config_path="config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"currency": "INR"}
```

---

## ⚡ Master Cheat Sheet

```python
# CLI & Configuration Master Cheat Sheet

import argparse, os, json
from dotenv import load_dotenv

# 1. Standard Argument Parser Setup
parser = argparse.ArgumentParser(prog="mytool", description="Developer Tool")
parser.add_argument("target", type=str, help="Target resource name")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logs")
parser.add_argument("-c", "--count", type=int, default=1, help="Repetition count")

# 2. .env File Pattern
load_dotenv()
db_uri = os.getenv("DATABASE_URL")

# 3. Running CLI with arguments via Terminal:
# python mytool.py user_data --verbose --count 5
```

---

## ⚠️ Common Pitfalls & Best Practices

1. **Hardcoding Secret Keys in Source Code:**
   - ❌ `API_KEY = "sk_live_123456789"` (Exposes secret credentials when pushing code to GitHub).
   - ✅ Always load secrets via `os.getenv("API_KEY")` and include `.env` in `.gitignore`.

2. **Omitting `dest` or `required=True` on Subparsers:**
   - ❌ `subparsers = parser.add_subparsers()` (If user runs command without subcommand, `args.command` evaluates to `None` without helpful error message).
   - ✅ `subparsers = parser.add_subparsers(dest="command", required=True)`.

---

## ❓ Practice & Interview Questions (With Solutions)

### Q1: What is the difference between Positional and Optional arguments in `argparse`?
**Answer:** Positional arguments are required by default and identified by their ordinal order in the command line (e.g. `python script.py value1`). Optional arguments are prefixed with `-` or `--` (e.g. `--output file.txt`), can be passed in any order, and can specify default values if omitted.

### Q2: Why is `.env.example` necessary in open-source Python repositories?
**Answer:** Real `.env` files containing production credentials are listed in `.gitignore` to prevent credential leakage. `.env.example` serves as a template documenting mandatory environment variables and key names required to configure the application locally.

---

## 📝 Recap Checklist
- [x] Used `sys.argv` and `argparse.ArgumentParser`.
- [x] Defined positional arguments, optional flags (`--verbose`), choices, and default values.
- [x] Implemented CLI subcommands using `add_subparsers()`.
- [x] Managed environment variables using `os.getenv()` and `python-dotenv`.
- [x] Created `.env` and `.env.example` configuration workflows.
- [x] Built CLI Expense Tracker, CLI GitHub Profile Tool, and CLI Study Tracker projects.
