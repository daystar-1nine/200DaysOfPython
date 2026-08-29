# ==============================================================================
# Program    : Task 7 - DELETE Request (task7_delete_request.py)
# Objective  : Execute DELETE request removing a target resource from remote server.
# Concept    : HTTP DELETE Method
# Why Used   : Deletes post entity record #1.
# ==============================================================================

import requests

def delete_post():
    url = "https://jsonplaceholder.typicode.com/posts/1"
    response = requests.delete(url, timeout=10)
    print(f"DELETE Status Code: {response.status_code}")
    print(f"Response Body     : {response.json()}")

if __name__ == "__main__":
    delete_post()
