# ==============================================================================
# Program    : Runnable Package Entry Point (__main__.py)
# Objective  : Allow direct module execution via `python -m expense_tracker`.
# Concept    : Package Execution Protocol
# Why Used   : Delegates execution to main.main().
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from expense_tracker.main import main

if __name__ == "__main__":
    main()
