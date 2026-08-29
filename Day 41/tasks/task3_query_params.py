# ==============================================================================
# Program    : Task 3 - Query Parameters (task3_query_params.py)
# Objective  : Execute GET request passing query parameter dictionary via params=.
# Concept    : HTTP URL Query Parameters
# Why Used   : Filters remote API server resources dynamically.
# ==============================================================================

import requests

def main():
    url = "https://jsonplaceholder.typicode.com/posts"
    # What is used : params dictionary parameter
    # Why it is used: Appends ?userId=1 to URL automatically
    params = {"userId": 1}

    response = requests.get(url, params=params, timeout=10)
    print(f"Generated Request URL: {response.url}")
    print(f"Status Code          : {response.status_code}")

    posts = response.json()
    print(f"Total Posts for User #1: {len(posts)}")

if __name__ == "__main__":
    main()
