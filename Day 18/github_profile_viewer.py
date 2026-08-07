# ==============================================================================
# Program    : GitHub Profile Viewer (Mini Project)
# Objective  : CLI application fetching GitHub profile details using REST API.
# Concept    : Consuming REST APIs & Structuring Parsed JSON Output
# Why Used   : Fetches user bio, followers, following, repos, and account creation date via requests.
# ==============================================================================

import requests

def fetch_github_profile(username):
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json(), False
        elif response.status_code == 404:
            print(f"Error: GitHub username '{username}' not found!")
            return None, False
        else:
            print(f"API Error (HTTP {response.status_code})")
            return None, False
    except requests.exceptions.RequestException:
        # Fallback Mock Data if offline / network error
        mock_data = {
            "login": username,
            "name": f"{username.capitalize()} (Offline Mode)",
            "bio": "Python Developer & Open Source Contributor",
            "public_repos": 42,
            "followers": 150,
            "following": 80,
            "created_at": "2023-01-15T00:00:00Z"
        }
        return mock_data, True

def sanitize_ascii(text):
    """Sanitizes text by removing non-ASCII characters for Windows CP1252 compatibility."""
    if not text:
        return "N/A"
    return text.encode("ascii", "ignore").decode("ascii")

def display_profile(profile, is_mock=False):
    print("\n==========================================================")
    print("                 GITHUB PROFILE CARD                      ")
    print("==========================================================")
    if is_mock:
        print("[NOTICE] Displaying Fallback Data (Offline/Rate Limited)")
    print(f"Username            : {profile.get('login')}")
    print(f"Full Name           : {sanitize_ascii(profile.get('name'))}")
    print(f"Bio                 : {sanitize_ascii(profile.get('bio'))}")
    print(f"Public Repositories : {profile.get('public_repos')}")
    print(f"Followers           : {profile.get('followers')}")
    print(f"Following           : {profile.get('following')}")
    created_date = profile.get('created_at', '')[:10]
    print(f"Account Created On  : {created_date}")
    print("==========================================================\n")

def main():
    print("=== GitHub Profile Viewer CLI ===")
    user_input = input("Enter GitHub Username (default 'daystar-1nine'): ").strip()
    username = user_input if user_input else "daystar-1nine"

    profile_data, is_mock = fetch_github_profile(username)
    if profile_data:
        display_profile(profile_data, is_mock)

if __name__ == "__main__":
    main()
