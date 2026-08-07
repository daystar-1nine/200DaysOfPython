# ==============================================================================
# Module     : GitHub API Helper Utility
# Objective  : Fetch GitHub user profile data using REST API requests.
# Concept    : Modular Function Extraction & API Consumption
# Why Used   : Encapsulates HTTP GET logic and error handling into reusable function.
# ==============================================================================

import requests

def fetch_user_details(username):
    """Fetches user details from GitHub REST API with fallback data for offline mode."""
    url = f"https://api.github.com/users/{username}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return res.json(), None
        elif res.status_code == 404:
            return None, f"User '{username}' not found on GitHub."
        else:
            return None, f"GitHub API Error (Status Code: {res.status_code})"
    except requests.exceptions.RequestException:
        # Fallback offline mock data
        mock = {
            "login": username,
            "name": f"{username.capitalize()} (Offline Data)",
            "bio": "Software Engineer & Open Source Developer",
            "public_repos": 35,
            "followers": 120,
            "following": 45
        }
        return mock, None
