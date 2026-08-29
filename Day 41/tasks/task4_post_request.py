# ==============================================================================
# Program    : Task 4 - POST Request (task4_post_request.py)
# Objective  : Execute POST request sending JSON payload to create new post resource.
# Concept    : HTTP POST Method & JSON Body Transmission
# Why Used   : Creates new record on remote server and expects 201 Created status code.
# ==============================================================================

import requests

def create_post():
    url = "https://jsonplaceholder.typicode.com/posts"
    # What is used : json= payload parameter
    # Why it is used: Automatically serializes dictionary to JSON and sets Content-Type header
    payload = {
        "title": "Learning Python Real-World Development",
        "body": "Day 41 HTTP & Real-World APIs",
        "userId": 1
    }

    response = requests.post(url, json=payload, timeout=10)
    print(f"Status Code   : {response.status_code}")  # Expected: 201 Created
    print(f"Created Record: {response.json()}")

if __name__ == "__main__":
    create_post()
