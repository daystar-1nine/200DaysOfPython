# ==============================================================================
# Program    : Application Configuration (config.py)
# Objective  : Centralize constants, default formats, and export paths.
# Concept    : Configuration Management
# Why Used   : Provides centralized configuration constants across modules.
# ==============================================================================

import os

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
