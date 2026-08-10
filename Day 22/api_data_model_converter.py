# ==============================================================================
# Program    : API Data Model Converter (Bonus Challenge)
# Objective  : Convert raw API JSON dictionary responses into type-safe Dataclass User instances.
# Concept    : Dataclass Classmethods & Deserialization
# Why Used   : Connects API payloads (Day 18) with type-safe Dataclass domain models (Day 22).
# ==============================================================================

from dataclasses import dataclass
import json

# API JSON response string payload
RAW_API_RESPONSE: str = """
[
    {"id": 101, "name": "Suraj Sawant", "email": "suraj@example.com", "role": "Admin"},
    {"id": 102, "name": "Rahul Sharma", "email": "rahul@example.com"},
    {"id": 103, "name": "Priya Patel", "email": "priya@example.com", "role": "Developer"}
]
"""

# What is used : User dataclass with optional role parameter
@dataclass
class User:
    id: int
    name: str
    email: str
    role: str | None = "Member"

    # What is used : Classmethod converter
    # Why it is used: Parses dictionary keys into Dataclass constructor parameters
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            id=data["id"],
            name=data["name"],
            email=data["email"],
            role=data.get("role", "Member")
        )

def main() -> None:
    print("==========================================================")
    print("           API JSON DATA MODEL CONVERTER                  ")
    print("==========================================================")

    raw_json_data: list[dict] = json.loads(RAW_API_RESPONSE)
    users: list[User] = [User.from_dict(item) for item in raw_json_data]

    print("\n--- Parsed Type-Safe User Dataclass Objects ---")
    for u in users:
        print(f"User Obj: {u}")
        print(f"  -> ID: {u.id} | Name: {u.name:<15} | Email: {u.email:<18} | Role: {u.role}")

if __name__ == "__main__":
    main()
