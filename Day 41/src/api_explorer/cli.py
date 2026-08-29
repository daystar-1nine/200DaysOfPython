# ==============================================================================
# Program    : REST API Explorer Interactive CLI Interface (cli.py)
# Objective  : Interactive Terminal Menu for List Users, Get User, Create/Update/Delete Post, Search.
# Concept    : Terminal User Interface & Menu Dispatching
# Why Used   : Interactive user interface for exploring remote REST APIs.
# ==============================================================================

import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.client import APIClient
from api_explorer.exceptions import APIError
from api_explorer.services import UserService, PostService

def print_menu():
    print("\n====================================")
    print("       REST API EXPLORER            ")
    print("====================================")
    print("1. List Users")
    print("2. Get User")
    print("3. Create Post")
    print("4. Update Post")
    print("5. Delete Post")
    print("6. Search Users")
    print("7. Exit")

def run_cli_menu(base_url: str = "https://jsonplaceholder.typicode.com"):
    client = APIClient(base_url=base_url)
    user_service = UserService(client)
    post_service = PostService(client)

    while True:
        print_menu()
        choice = input("\nSelect Option (1-7): ").strip()

        if choice == "1":
            try:
                users = user_service.list_users()
                print("\nUsers")
                print("──────────────────────────────────────────────────")
                print(f"{'ID':<6} {'NAME':<25} {'EMAIL'}")
                print("──────────────────────────────────────────────────")
                for u in users:
                    print(f"{u.id:<6} {u.name:<25} {u.email}")
            except APIError as e:
                print(f"\nAPI Error: {e}")

        elif choice == "2":
            uid_str = input("Enter User ID: ").strip()
            if not uid_str.isdigit():
                print("Invalid User ID input.")
                continue
            try:
                user = user_service.get_user(int(uid_str))
                print(f"\n{user}")
            except APIError as e:
                print(f"\nAPI Error: {e}")

        elif choice == "3":
            title = input("Enter Post Title: ").strip()
            body = input("Enter Post Body: ").strip()
            uid_str = input("Enter Author User ID: ").strip()
            if not uid_str.isdigit():
                print("Invalid User ID input.")
                continue
            try:
                post = post_service.create_post(title=title, body=body, user_id=int(uid_str))
                print(f"\nPost Created Successfully!\n{post}")
            except APIError as e:
                print(f"\nAPI Error: {e}")

        elif choice == "4":
            pid_str = input("Enter Post ID to Update: ").strip()
            title = input("Enter New Post Title: ").strip()
            body = input("Enter New Post Body: ").strip()
            uid_str = input("Enter User ID: ").strip()
            if not pid_str.isdigit() or not uid_str.isdigit():
                print("Invalid ID input.")
                continue
            try:
                post = post_service.update_post(post_id=int(pid_str), title=title, body=body, user_id=int(uid_str))
                print(f"\nPost Updated Successfully!\n{post}")
            except APIError as e:
                print(f"\nAPI Error: {e}")

        elif choice == "5":
            pid_str = input("Enter Post ID to Delete: ").strip()
            if not pid_str.isdigit():
                print("Invalid Post ID input.")
                continue
            try:
                post_service.delete_post(int(pid_str))
                print(f"\nPost #{pid_str} Deleted Successfully!")
            except APIError as e:
                print(f"\nAPI Error: {e}")

        elif choice == "6":
            query = input("Enter Search Name/Email Keyword: ").strip()
            try:
                matches = user_service.search_users(query)
                print(f"\nFound {len(matches)} Matching Users:\n")
                for u in matches:
                    print(f"ID #{u.id}: {u.name} ({u.email}) - {u.company}")
            except APIError as e:
                print(f"\nAPI Error: {e}")

        elif choice == "7":
            print("\nExiting REST API Explorer. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1-7.")
