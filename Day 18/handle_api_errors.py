# ==============================================================================
# Program    : Robust API Error Handling
# Objective  : Handle HTTP request exceptions cleanly using try-except blocks.
# Concept    : Exception Handling with requests.exceptions.RequestException
# Why Used   : Prevents application crashes when encountering network timeouts or 4xx/5xx HTTP errors.
# ==============================================================================

import requests

def safe_fetch(url):
    print(f"Attempting GET request to '{url}'...")
    try:
        # What is used : requests.get() with timeout
        response = requests.get(url, timeout=3)
        
        # What is used : response.raise_for_status()
        # Why it is used: Raises HTTPError if status code is 4xx or 5xx
        response.raise_for_status()
        
        print(f"[SUCCESS] HTTP {response.status_code} - Fetched successfully!")
        return response.json()

    except requests.exceptions.HTTPError as http_err:
        print(f"[HTTP ERROR] Server responded with error status: {http_err}")
    except requests.exceptions.ConnectionError:
        print("[CONNECTION ERROR] Failed to connect to server. Check internet connection.")
    except requests.exceptions.Timeout:
        print("[TIMEOUT ERROR] Request timed out!")
    except requests.exceptions.RequestException as err:
        print(f"[ERROR] An unexpected error occurred: {err}")
    return None

def main():
    print("=== Robust API Error Handling ===")
    safe_fetch("https://api.github.com/users/octocat")
    safe_fetch("https://api.github.com/invalid_route_404")

if __name__ == "__main__":
    main()
