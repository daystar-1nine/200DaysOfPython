# ==============================================================================
# Program    : Exercise 1 - User Dependency Injection (ex1_user_injection.py)
# Objective  : Inject get_current_user dependency into a FastAPI endpoint.
# Concept    : Basic Dependency Injection via Depends()
# Why Used   : Demonstrates injecting user details into route parameters automatically.
# ==============================================================================

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

app = FastAPI(title="Exercise 1 - User Injection")

# What is used : Dependency Callable
# Why it is used: Resolves current user dictionary payload
def get_current_user() -> dict:
    return {"id": 1, "name": "Suraj Sawant", "role": "admin"}

# What is used : Depends(get_current_user)
# Why it is used: Injects resolved current user object into user parameter
@app.get("/me")
def read_me(user: dict = Depends(get_current_user)):
    return {"status": "authenticated", "user": user}

if __name__ == "__main__":
    client = TestClient(app)
    res = client.get("/me")
    print(f"Exercise 1 Response: {res.json()}")
