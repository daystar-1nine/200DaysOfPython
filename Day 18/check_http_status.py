# ==============================================================================
# Program    : Check and Interpret HTTP Status Codes
# Objective  : Demonstrate evaluating HTTP status codes returned by REST API responses.
# Concept    : HTTP Status Code Handling (200, 404, 500, etc.)
# Why Used   : Evaluates response status to determine whether request succeeded or failed.
# ==============================================================================

import requests

status_codes_map = {
    200: "200 OK - Success!",
    201: "201 Created - Resource created successfully",
    400: "400 Bad Request - Invalid request payload/parameters",
    401: "401 Unauthorized - Authentication credentials missing",
    403: "403 Forbidden - Insufficient permission rights",
    404: "404 Not Found - Resource endpoint does not exist",
    500: "500 Internal Server Error - Remote server failure"
}

def inspect_endpoint(url):
    try:
        res = requests.get(url, timeout=5)
        code = res.status_code
        description = status_codes_map.get(code, f"Status Code: {code}")
        print(f"URL: {url:<40} -> {description}")
    except requests.exceptions.RequestException as e:
        print(f"URL: {url:<40} -> Request Exception: {e}")

def main():
    print("=== Checking HTTP Status Codes ===")
    inspect_endpoint("https://api.github.com")
    inspect_endpoint("https://api.github.com/non_existent_endpoint_12345")

if __name__ == "__main__":
    main()
