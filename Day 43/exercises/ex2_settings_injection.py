# ==============================================================================
# Program    : Exercise 2 - Settings Dependency Injection (ex2_settings_injection.py)
# Objective  : Inject configuration settings dependency into a FastAPI endpoint.
# Concept    : Configuration Injection via Depends()
# Why Used   : Supplies application config dict to endpoints without global variable clutter.
# ==============================================================================

from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

app = FastAPI(title="Exercise 2 - Settings Injection")

# What is used : Settings Dependency Callable
# Why it is used: Resolves app environment configuration settings
def get_settings() -> dict:
    return {
        "app_name": "User Management API V2",
        "environment": "development",
        "debug": True
    }

# What is used : Depends(get_settings)
# Why it is used: Injects settings dictionary into endpoint
@app.get("/config")
def read_config(settings: dict = Depends(get_settings)):
    return settings

if __name__ == "__main__":
    client = TestClient(app)
    res = client.get("/config")
    print(f"Exercise 2 Response: {res.json()}")
