# ==============================================================================
# Program    : GitHub User Finder Application
# Objective  : Entry point CLI querying user profile data using github helper module.
# Concept    : Modular Project Architecture & Imports
# Why Used   : Connects utils/github.py module to CLI user interface.
# ==============================================================================

import os
import sys

# What is used : Appending current directory to sys.path for clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.github import fetch_user_details

def sanitize_ascii(text):
    """Sanitizes text by removing non-ASCII characters for Windows CP1252 compatibility."""
    if not text:
        return "N/A"
    return text.encode("ascii", "ignore").decode("ascii")

def main():
    print("==========================================================")
    print("                GITHUB USER FINDER APP                    ")
    print("==========================================================")
    
    input_user = input("Enter GitHub Username (default 'daystar-1nine'): ").strip()
    username = input_user if input_user else "daystar-1nine"

    data, err = fetch_user_details(username)
    if err:
        print(f"\n[ERROR] {err}")
    elif data:
        print("\n---------------- USER PROFILE DETAILS ----------------")
        print(f"Username            : {data.get('login')}")
        print(f"Name                : {sanitize_ascii(data.get('name'))}")
        print(f"Public Repositories : {data.get('public_repos')}")
        print(f"Followers           : {data.get('followers')}")
        print(f"Following           : {data.get('following')}")
        print("------------------------------------------------------\n")

if __name__ == "__main__":
    main()
