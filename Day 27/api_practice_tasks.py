# ==============================================================================
# Program    : Real-World API Practice Tasks (Tasks 1 to 8)
# Objective  : Demonstrate GET, POST, Query Params, Timeouts, Retries, Logging, and .env loading.
# Concept    : requests Module, Resilience, Exponential Backoff & Security
# Why Used   : Complete walkthrough covering all 8 practice tasks in Day 27 requirements.
# ==============================================================================

import logging
import os
import time
import requests
from dotenv import load_dotenv

# Task 7: Setup logging
LOG_FILE = "api_practice.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("APIPractice")

# Task 6: Load environment variables
load_dotenv()
API_TOKEN = os.getenv("GITHUB_TOKEN", "mock_token_secret_123")

def task1_get_request() -> None:
    """Task 1 & Task 5: Send GET request with timeout and print status code + JSON."""
    url = "https://jsonplaceholder.typicode.com/posts/1"
    print("\n--- [Task 1 & 5] GET REQUEST WITH TIMEOUT ---")
    try:
        # What is used : timeout=3
        # Why it is used: Ensures request does not hang process if server is unresponsive
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        logger.info("Task 1 GET Request status: %d", resp.status_code)
        print(f"Status Code : {resp.status_code}")
        print("Response JSON:", resp.json())
    except requests.RequestException as e:
        logger.error("Task 1 GET Request failed: %s", e)

def task2_query_parameters() -> None:
    """Task 2: Use query parameters in GET request."""
    url = "https://jsonplaceholder.typicode.com/comments"
    # What is used : params dictionary
    # Why it is used: Encodes /comments?postId=1 safely
    params = {"postId": 1}
    print("\n--- [Task 2] GET REQUEST WITH QUERY PARAMETERS ---")
    try:
        resp = requests.get(url, params=params, timeout=3)
        resp.raise_for_status()
        print(f"URL Requested : {resp.url}")
        print(f"Total Comments Returned: {len(resp.json())}")
    except requests.RequestException as e:
        logger.error("Task 2 Query Params request failed: %s", e)

def task3_post_request() -> None:
    """Task 3: Send POST request with JSON payload."""
    url = "https://jsonplaceholder.typicode.com/posts"
    payload = {
        "title": "200 Days of Python",
        "body": "Mastering Real-World APIs",
        "userId": 19
    }
    print("\n--- [Task 3] POST REQUEST WITH JSON PAYLOAD ---")
    try:
        # What is used : json=payload
        # Why it is used: Serializes payload dict to JSON and sets Content-Type: application/json
        resp = requests.post(url, json=payload, timeout=3)
        resp.raise_for_status()
        print(f"Status Code : {resp.status_code}")
        print("Created Post Response:", resp.json())
    except requests.RequestException as e:
        logger.error("Task 3 POST Request failed: %s", e)

def task4_exception_handling() -> None:
    """Task 4: Demonstrate handling Timeout, ConnectionError, and HTTPError."""
    print("\n--- [Task 4] EXCEPTION HANDLING DEMO ---")
    
    # Invalid URL to trigger HTTPError (404)
    invalid_url = "https://jsonplaceholder.typicode.com/invalid_endpoint_404"
    try:
        resp = requests.get(invalid_url, timeout=3)
        resp.raise_for_status()
    except requests.Timeout:
        print("[CAUGHT] Timeout Exception")
    except requests.ConnectionError:
        print("[CAUGHT] ConnectionError Exception")
    except requests.HTTPError as e:
        print(f"[CAUGHT] HTTPError Exception (404 Not Found): {e.response.status_code}")
        logger.error("HTTPError caught: %s", e)
    except requests.RequestException as e:
        print(f"[CAUGHT] RequestException: {e}")

def task8_retry_exponential_backoff() -> None:
    """Task 8: Implement retry mechanism with exponential backoff."""
    url = "https://httpbin.org/status/503"  # Endpoint returning 503 Service Unavailable
    max_retries = 2
    print("\n--- [Task 8] RETRY MECHANISM WITH EXPONENTIAL BACKOFF ---")
    
    for attempt in range(max_retries):
        try:
            print(f"--> Attempt {attempt + 1}/{max_retries}: Requesting {url}...")
            resp = requests.get(url, timeout=2)
            resp.raise_for_status()
            print("Request Succeeded!")
            break
        except requests.RequestException as e:
            logger.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                print(f"<-- All {max_retries} retry attempts failed. Gracefully caught exception.")
            else:
                backoff = 0.1 * (2 ** attempt)
                print(f"    Failed. Retrying in {backoff:.2f} seconds (Exponential Backoff)...")
                time.sleep(backoff)

def main() -> None:
    print("=== DAY 27: REAL-WORLD API PRACTICE TASKS 1 TO 8 ===")
    task1_get_request()
    task2_query_parameters()
    task3_post_request()
    task4_exception_handling()
    task8_retry_exponential_backoff()
    print("\nAll practice tasks executed successfully. Logs recorded in 'api_practice.log'.")

    # Cleanup log file
    if os.path.exists(LOG_FILE):
        try:
            os.remove(LOG_FILE)
        except OSError:
            pass

if __name__ == "__main__":
    main()
