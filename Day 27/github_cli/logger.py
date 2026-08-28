# ==============================================================================
# Program    : GitHub CLI Logger Factory
# Objective  : Provide modular application logger writing API operations to log file.
# Concept    : Modular Logging Architecture
# Why Used   : Formats and logs API HTTP requests, retries, and errors.
# ==============================================================================

import logging
import os

LOG_FILE: str = os.path.join(os.path.dirname(__file__), "..", "github_api.log")

def get_logger(name: str = "GitHubCLI") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
