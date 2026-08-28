# ==============================================================================
# Program    : PyFinance Runnable Package Entry Point
# Objective  : Allow execution via `python -m pyfinance`.
# Concept    : Module Entry Point Protocol
# Why Used   : Boots main application entry point.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.main import main

if __name__ == "__main__":
    main()
