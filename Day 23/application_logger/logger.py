# ==============================================================================
# Module     : Modular Application Logger Factory (Mini Project)
# Objective  : Reusable logger factory returning configured Logger instances.
# Concept    : Modular Logger Architecture (getLogger, FileHandler, Formatter)
# Why Used   : Encapsulates logger creation logic for reuse across multiple application modules.
# ==============================================================================

import logging
import os

def get_logger(name: str = "app_logger", log_filename: str = "app.log") -> logging.Logger:
    """Returns a configured Logger with FileHandler and StreamHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid attaching duplicate handlers if logger already exists
    if not logger.handlers:
        # File Handler (INFO level and above)
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Stream Handler (DEBUG level and above for terminal)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)

        # Custom Formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
