# ==============================================================================
# Program    : Application Configuration Settings (config.py)
# Objective  : Provide central settings dictionary and get_settings dependency function.
# Concept    : Configuration Management & Dependency Provider
# Why Used   : Supplies configuration parameters via FastAPI Depends(get_settings).
# ==============================================================================

def get_settings() -> dict:
    """Configuration dependency function returning application settings dict."""
    return {
        "app_title": "User Management API V2",
        "version": "2.0.0",
        "environment": "development",
        "debug": True
    }
