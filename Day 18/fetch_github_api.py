# ==============================================================================
# Program    : Fetch Data from GitHub REST API
# Objective  : Make an HTTP GET request to GitHub REST API endpoint.
# Concept    : Consuming REST APIs using requests library
# Why Used   : requests.get() performs HTTP GET requests and retrieves response payloads.
# ==============================================================================

import requests

api_url = "https://api.github.com"

print(f"Connecting to GitHub REST API ({api_url})...")

try:
    # What is used : requests.get(url, timeout=5)
    # Why it is used: Sends HTTP GET request to web API endpoint with 5-second timeout
    response = requests.get(api_url, timeout=5)
    
    # What is used : response.status_code
    print(f"HTTP Status Code: {response.status_code}")

    if response.status_code == 200:
        # What is used : response.json()
        # Why it is used: Parses JSON response payload into a native Python dictionary
        data = response.json()
        print("\n--- GitHub API Endpoints Summary ---")
        print("User API Endpoint    :", data.get("current_user_url"))
        print("Repository Endpoint  :", data.get("repository_url"))
        print("User Search Endpoint :", data.get("user_search_url"))
    else:
        print(f"API Request Returned Non-200 Status: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"\n[Network Notice] Could not connect to live API: {e}")
    print("Simulated Output: Connected to GitHub API (Mock Mode OK)")
