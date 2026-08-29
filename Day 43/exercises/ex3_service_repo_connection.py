# ==============================================================================
# Program    : Exercise 3 - Service & Repository Connection (ex3_service_repo_connection.py)
# Objective  : Connect UserRepository and UserService using Dependency Injection.
# Concept    : Layered Architecture Dependency Chain
# Why Used   : Demonstrates chaining Repository -> Service -> Route via Depends().
# ==============================================================================

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

class UserRepository:
    def get_all(self):
        return [{"id": 1, "name": "Suraj Sawant"}]

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    def list_users(self):
        return self.repo.get_all()

# Dependency factories
def get_repo():
    return UserRepository()

def get_service(repo: UserRepository = Depends(get_repo)):
    return UserService(repo)

app = FastAPI(title="Exercise 3 - Service Repo Connection")

@app.get("/users")
def get_users(service: UserService = Depends(get_service)):
    return service.list_users()

if __name__ == "__main__":
    client = TestClient(app)
    res = client.get("/users")
    print(f"Exercise 3 Response: {res.json()}")
