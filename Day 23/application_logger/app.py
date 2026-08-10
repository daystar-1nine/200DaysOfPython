# ==============================================================================
# Program    : Application Logger Entry Point (Mini Project)
# Objective  : Demonstrate using the reusable get_logger factory in an application module.
# Concept    : Modular Logger Integration
# Why Used   : Connects logger.py to main application execution flow.
# ==============================================================================

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logger import get_logger

def main():
    logger = get_logger("MainApp", "app.log")
    
    print("==========================================================")
    print("              REUSABLE APPLICATION LOGGER                 ")
    print("==========================================================")

    logger.info("Application starting up...")
    logger.debug("Debugging configuration parameters: DEBUG=True")

    # Simulate operation
    try:
        logger.info("Processing user transaction...")
        val = 100 / 2
        logger.info("Transaction processed successfully: result=%.2f", val)

        logger.warning("Storage disk usage reached 82%")
        
        # Simulate handled error
        raise FileNotFoundError("config_settings.json missing")

    except FileNotFoundError as e:
        logger.error("Configuration error encountered: %s", e, exc_info=True)

    logger.info("Application shutting down gracefully.")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
