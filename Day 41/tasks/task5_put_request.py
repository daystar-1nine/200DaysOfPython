# ==============================================================================
# Program    : Task 5 - PUT Request (task5_put_request.py)
# Objective  : Execute PUT request replacing existing resource completely.
# Concept    : HTTP PUT Method (Full Resource Replacement)
# Why Used   : Replaces entire post entity record #1 on remote server.
# ==============================================================================

import requests

def update_entire_post():
    url = "https://jsonplaceholder.typicode.com/posts/1"
    payload = {
        "id": 1,
        "title": "Completely Updated Title",
        "body": "Completely replaced post body text.",
        "userId": 1
    }

    response = requests.put(url, json=payload, timeout=10)
    print(f"PUT Status Code: {response.status_code}")
    print(f"Replaced Object: {response.json()}")

if __name__ == "__main__":
    update_entire_post()
