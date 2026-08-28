# ==============================================================================
# Program    : GitHub CLI Main Entry Point (Main Project & Bonus Challenge)
# Objective  : Command-line interface supporting user, repos, search, and repo commands.
# Concept    : argparse CLI Subcommands & Real-World REST API Integration
# Why Used   : Parses terminal subcommands and displays styled GitHub profile and repository reports.
# ==============================================================================

import argparse
import json
import os
import sys

# Append module path for clean relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from github_api import GitHubService

def sanitize(text: str | None) -> str:
    """Sanitizes non-ASCII strings for Windows CP1252 terminal compatibility."""
    if not text:
        return "N/A"
    return text.encode("ascii", "ignore").decode("ascii")

def handle_user(service: GitHubService, username: str) -> None:
    profile = service.get_user(username)
    print("\n+------------------------------------+")
    print("|          GITHUB PROFILE            |")
    print("+------------------------------------+")
    print(f"Username     : {sanitize(profile.login)}")
    print(f"Name         : {sanitize(profile.name)}")
    print(f"Followers    : {profile.followers}")
    print(f"Following    : {profile.following}")
    print(f"Repositories : {profile.public_repos}")
    print(f"Bio          : {sanitize(profile.bio)}")
    print(f"Profile URL  : {profile.html_url}")
    print("+------------------------------------+\n")

def handle_repos(service: GitHubService, username: str) -> None:
    repos = service.get_repositories(username)
    print(f"\n------ REPOSITORIES ({sanitize(username)}) ------")
    print(f"{'Name':<25} {'Stars':<8} {'Forks':<8} {'Language':<12}")
    print("-" * 55)
    for r in repos:
        print(f"{sanitize(r.name):<25} {r.stars:<8} {r.forks:<8} {sanitize(r.language):<12}")
    print("-" * 55 + "\n")

def handle_search(service: GitHubService, query: str) -> None:
    results = service.search_repositories(query)
    print(f"\n------ SEARCH RESULTS FOR '{sanitize(query)}' ------")
    for idx, r in enumerate(results, start=1):
        print(f"{idx}. {sanitize(r.full_name):<30} | Stars: {r.stars:<6} | Lang: {sanitize(r.language)}")
    print("-------------------------------------------\n")

def handle_repo_detail(service: GitHubService, owner: str, repo: str, as_json: bool) -> None:
    repo_obj, raw_json = service.get_repository_details(owner, repo)
    if as_json:
        print(json.dumps(raw_json, indent=4))
        return

    print("\n------ REPOSITORY DETAILS ------")
    print(f"Name        : {sanitize(repo_obj.name)}")
    print(f"Owner       : {sanitize(repo_obj.owner)}")
    print(f"Stars       : {repo_obj.stars}")
    print(f"Forks       : {repo_obj.forks}")
    print(f"Issues      : {repo_obj.open_issues}")
    print(f"Language    : {sanitize(repo_obj.language)}")
    print(f"Created     : {repo_obj.created_at}")
    print(f"Updated     : {repo_obj.updated_at}")
    print(f"URL         : {repo_obj.html_url}")
    print("--------------------------------\n")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional GitHub API CLI Tool")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Command 1: user
    u_p = subparsers.add_parser("user", help="Fetch GitHub user profile")
    u_p.add_argument("username", type=str, help="GitHub username")

    # Command 2: repos
    r_p = subparsers.add_parser("repos", help="Fetch user public repositories")
    r_p.add_argument("username", type=str, help="GitHub username")

    # Command 3: search
    s_p = subparsers.add_parser("search", help="Search repositories")
    s_p.add_argument("query", type=str, help="Search query string")

    # Command 4: repo (Bonus Challenge)
    rd_p = subparsers.add_parser("repo", help="Fetch single repository details")
    rd_p.add_argument("owner", type=str, help="Repository owner")
    rd_p.add_argument("repository", type=str, help="Repository name")
    rd_p.add_argument("--json", action="store_true", help="Output raw JSON response")

    return parser

def main() -> None:
    print("=== MAIN PROJECT: GITHUB CLI APP ===")
    service = GitHubService()
    parser = create_parser()

    if len(sys.argv) == 1:
        print("Simulating CLI subcommand calls:\n")
        # 1. user
        handle_user(service, "octocat")
        # 2. repos
        handle_repos(service, "octocat")
        # 3. search
        handle_search(service, "python")
        # 4. repo detail
        handle_repo_detail(service, "octocat", "Hello-World", as_json=False)
    else:
        args = parser.parse_args()
        if args.command == "user":
            handle_user(service, args.username)
        elif args.command == "repos":
            handle_repos(service, args.username)
        elif args.command == "search":
            handle_search(service, args.query)
        elif args.command == "repo":
            handle_repo_detail(service, args.owner, args.repository, args.json)

if __name__ == "__main__":
    main()
