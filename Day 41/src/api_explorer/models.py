# ==============================================================================
# Program    : API Response Domain Models (models.py)
# Objective  : User and Post dataclasses with Dunder Methods (__str__, __repr__, __getitem__, __len__).
# Concept    : Dataclasses (Day 39) & Dunder Method Protocols (Day 32)
# Why Used   : Provides printable, subscriptable, and printable models for API responses.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class User:
    """User response model dataclass with Dunder Protocols."""
    id: int
    name: str
    username: str
    email: str
    city: str = ""
    company: str = ""
    phone: str = ""
    website: str = ""

    # What is used : __str__ dunder method (Day 32 requirement)
    # Why it is used: Formats clean ASCII printable user card
    def __str__(self) -> str:
        return (
            f"USER DETAILS #{self.id}\n"
            f"────────────────────\n"
            f"Name     : {self.name}\n"
            f"Username : {self.username}\n"
            f"Email    : {self.email}\n"
            f"City     : {self.city}\n"
            f"Company  : {self.company}\n"
            f"Phone    : {self.phone}\n"
            f"Website  : {self.website}"
        )

    # What is used : __repr__ dunder method
    # Why it is used: Provides developer-friendly representation
    def __repr__(self) -> str:
        return f"<User id={self.id} name='{self.name}' email='{self.email}'>"

    # What is used : __getitem__ dunder method
    # Why it is used: Enables dictionary-style subscripting (e.g. user['name'])
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """Factory method constructing User instance from API response dict."""
        address = data.get("address", {})
        company = data.get("company", {})
        city = address.get("city", "") if isinstance(address, dict) else ""
        comp_name = company.get("name", "") if isinstance(company, dict) else ""

        return cls(
            id=int(data.get("id", 0)),
            name=str(data.get("name", "")),
            username=str(data.get("username", "")),
            email=str(data.get("email", "")),
            city=city,
            company=comp_name,
            phone=str(data.get("phone", "")),
            website=str(data.get("website", ""))
        )

@dataclass(slots=True)
class Post:
    """Post response model dataclass with Dunder Protocols."""
    id: int
    title: str
    body: str
    user_id: int = 0

    # What is used : __str__ dunder method (Day 32 requirement)
    # Why it is used: Formats clean ASCII printable post card
    def __str__(self) -> str:
        return (
            f"POST #{self.id} (User #{self.user_id})\n"
            f"──────────────────────────────────────────────────\n"
            f"Title : {self.title}\n"
            f"Body  : {self.body}"
        )

    def __repr__(self) -> str:
        return f"<Post id={self.id} title='{self.title[:20]}...' user_id={self.user_id}>"

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Post":
        """Factory method constructing Post instance from API response dict."""
        return cls(
            id=int(data.get("id", 0)),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            user_id=int(data.get("userId", data.get("user_id", 0)))
        )
