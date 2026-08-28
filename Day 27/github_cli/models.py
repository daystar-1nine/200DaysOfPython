# ==============================================================================
# Program    : GitHub CLI Data Models
# Objective  : Define type-safe Dataclass schemas for User Profiles and Repositories.
# Concept    : Dataclasses & Deserialization
# Why Used   : Converts JSON API dictionaries into type-safe domain objects.
# ==============================================================================

from dataclasses import dataclass

@dataclass
class UserProfile:
    login: str
    name: str
    followers: int
    following: int
    public_repos: int
    bio: str
    html_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            login=data.get("login", "N/A"),
            name=data.get("name") or data.get("login", "N/A"),
            followers=data.get("followers", 0),
            following=data.get("following", 0),
            public_repos=data.get("public_repos", 0),
            bio=data.get("bio") or "No bio provided",
            html_url=data.get("html_url", "")
        )

@dataclass
class Repository:
    name: str
    full_name: str
    owner: str
    stars: int
    forks: int
    open_issues: int
    language: str
    created_at: str
    updated_at: str
    html_url: str

    @classmethod
    def from_dict(cls, data: dict) -> "Repository":
        return cls(
            name=data.get("name", "N/A"),
            full_name=data.get("full_name", "N/A"),
            owner=data.get("owner", {}).get("login", "N/A") if isinstance(data.get("owner"), dict) else "N/A",
            stars=data.get("stargazers_count", 0),
            forks=data.get("forks_count", 0),
            open_issues=data.get("open_issues_count", 0),
            language=data.get("language") or "Unknown",
            created_at=data.get("created_at", "N/A"),
            updated_at=data.get("updated_at", "N/A"),
            html_url=data.get("html_url", "")
        )
