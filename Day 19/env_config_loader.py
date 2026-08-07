# ==============================================================================
# Program    : Environment Configuration Loader (.env)
# Objective  : Demonstrate reading environment variables securely from .env files.
# Concept    : Configuration Secrets Isolation (os.getenv & dotenv)
# Why Used   : Keeps API keys and secrets outside git repository source code.
# ==============================================================================

import os

# Custom fallback parser if python-dotenv is not installed
def load_env_file(filepath=".env"):
    """Reads .env file line by line and populates os.environ."""
    if not os.path.exists(filepath):
        print(f"[Notice] '.env' file not found at '{filepath}'. Using default environment fallbacks.")
        return False
    
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")
    return True

def main():
    print("=== ENVIRONMENT CONFIGURATION LOADER ===")
    
    # Try importing python-dotenv if available, else use custom parser
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("Loaded environment via python-dotenv package.")
    except ImportError:
        load_env_file()
        print("Loaded environment via custom fallback parser.")

    # Retrieve environment variables safely
    api_key = os.getenv("API_KEY", "default_demo_api_key_12345")
    secret_key = os.getenv("SECRET_KEY", "demo_secret_key")
    debug_flag = os.getenv("DEBUG", "True")

    print("\n--- Loaded Environment Configuration ---")
    print(f"API_KEY    : {api_key}")
    print(f"SECRET_KEY : {secret_key}")
    print(f"DEBUG      : {debug_flag}")

if __name__ == "__main__":
    main()
