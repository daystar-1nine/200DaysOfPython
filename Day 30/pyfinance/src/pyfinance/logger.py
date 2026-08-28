# ==============================================================================
# Program    : PyFinance Logger Factory
# Objective  : Provide application logging writing operational events to logs/pyfinance.log.
# Concept    : Modular Logging
# Why Used   : Records application events, warnings, and error tracebacks safely.
# ==============================================================================

import logging
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.config import LOG_FILE_PATH, LOG_LEVEL

def get_logger(name: str = "PyFinance") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
