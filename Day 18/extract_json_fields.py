# ==============================================================================
# Program    : Extract Selected Fields from Nested JSON Payload
# Objective  : Parse complex nested JSON responses and extract specific data fields.
# Concept    : Nested Dictionary & List Traversal
# Why Used   : Extracts key attributes from deeply nested API payload structures.
# ==============================================================================

import json

# Sample API JSON response payload
api_json_response = """
{
    "status": "success",
    "data": {
        "user_id": 9876,
        "profile": {
            "first_name": "Suraj",
            "last_name": "Sawant",
            "email": "suraj@example.com"
        },
        "repositories": [
            {"name": "200DaysOfPython", "stars": 150, "language": "Python"},
            {"name": "AI-Agent-Framework", "stars": 320, "language": "Python"}
        ]
    }
}
"""

def main():
    print("=== Extracting Specific Fields from JSON Payload ===")
    
    # What is used : json.loads()
    payload = json.loads(api_json_response)

    # What is used : Dictionary indexing and list iteration
    user_id = payload["data"]["user_id"]
    full_name = f"{payload['data']['profile']['first_name']} {payload['data']['profile']['last_name']}"
    email = payload["data"]["profile"]["email"]

    print(f"User ID   : {user_id}")
    print(f"Full Name : {full_name}")
    print(f"Email     : {email}")

    print("\nUser Repositories:")
    for repo in payload["data"]["repositories"]:
        print(f"  - {repo['name']:<20} | Stars: {repo['stars']:<4} | Language: {repo['language']}")

if __name__ == "__main__":
    main()
