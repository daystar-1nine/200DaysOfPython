# ==============================================================================
# Program    : Task 6 - PATCH Request (task6_patch_request.py)
# Objective  : Execute PATCH request updating specific attribute of a resource partially.
# Concept    : HTTP PATCH Method (Partial Attribute Modification)
# Why Used   : Modifies only the 'title' field without re-sending entire object schema.
# ==============================================================================

import requests

def patch_post_title():
    url = "https://jsonplaceholder.typicode.com/posts/1"
    payload = {
        "title": "Partially Modified Title via PATCH"
    }

    response = requests.patch(url, json=payload, timeout=10)
    print(f"PATCH Status Code: {response.status_code}")
    print(f"Patched Object   : {response.json()}")

if __name__ == "__main__":
    patch_post_title()
