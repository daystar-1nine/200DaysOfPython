# ==============================================================================
# Program    : Expense Tracker Centralized Configuration
# Objective  : Load application config and environment settings.
# Concept    : Configuration Management
# Why Used   : Prevents hardcoding database paths and log levels in code.
# ==============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_PATH: str = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "..", "..", "expenses.db"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
