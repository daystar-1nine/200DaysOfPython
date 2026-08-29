# ==============================================================================
# Program    : Task 1 - First API Request (task1_first_request.py)
# Objective  : Execute GET request to public JSONPlaceholder API and inspect response.
# Concept    : Basic HTTP GET Request using requests
# Why Used   : Demonstrates fetching external remote data via HTTP protocol.
# ==============================================================================

import requests

def main():
    url = "https://jsonplaceholder.typicode.com/users"
    print(f"Executing GET Request to: {url}")

    # What is used : requests.get()
    # Why it is used: Sends HTTP GET request to retrieve user dataset
    response = requests.get(url, timeout=10)

    print(f"Status Code: {response.status_code}")
    users = response.json()
    print(f"Response Data Type: {type(users)}")
    print(f"Total Users Count : {len(users)}")
    print(f"First User Record : {users[0]['name']} ({users[0]['email']})")

if __name__ == "__main__":
    main()
