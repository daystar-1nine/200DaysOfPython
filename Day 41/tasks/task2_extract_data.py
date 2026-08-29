# ==============================================================================
# Program    : Task 2 - Nested Data Extraction (task2_extract_data.py)
# Objective  : Extract specific fields (Name, Email, City, Company) from API JSON.
# Concept    : Nested Dictionary & List Traversal
# Why Used   : Practice navigating complex structured API responses.
# ==============================================================================

import requests

def extract_user_details():
    url = "https://jsonplaceholder.typicode.com/users"
    response = requests.get(url, timeout=10)
    users = response.json()

    print("==================================================")
    print("           EXTRACTED USER DETAILS                ")
    print("==================================================\n")

    for u in users[:3]:  # Print first 3 users
        # What is used : Nested dictionary key lookup
        # Why it is used: Accesses nested 'address' -> 'city' and 'company' -> 'name'
        name = u.get("name")
        email = u.get("email")
        city = u.get("address", {}).get("city")
        company = u.get("company", {}).get("name")

        print(f"Name   : {name}")
        print(f"Email  : {email}")
        print(f"City   : {city}")
        print(f"Company: {company}\n")

if __name__ == "__main__":
    extract_user_details()
