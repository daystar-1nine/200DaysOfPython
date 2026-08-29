# ==============================================================================
# Program    : Task 8 - HTTP Request Exception Handling (task8_error_handling.py)
# Objective  : Safely handle timeouts, 404 Not Found, and network connection errors.
# Concept    : HTTP Request Error Boundaries & response.raise_for_status()
# Why Used   : Prevents unhandled crashes during HTTP communication failures.
# ==============================================================================

import sys
import requests

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def safe_api_call(url: str):
    try:
        # What is used : timeout=5 and response.raise_for_status()
        # Why it is used: Catches connection hangs and HTTP 4xx/5xx status codes
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print(f"Success [{response.status_code}]: {len(response.json())} items received.")
    except requests.exceptions.Timeout:
        print("Error: Request timed out after 5 seconds.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error [{response.status_code}]: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Network Request Exception: {e}")

if __name__ == "__main__":
    print("--- 1. Testing Valid Request ---")
    safe_api_call("https://jsonplaceholder.typicode.com/users")

    print("\n--- 2. Testing 404 Not Found Endpoint ---")
    safe_api_call("https://jsonplaceholder.typicode.com/invalid_endpoint_999")
