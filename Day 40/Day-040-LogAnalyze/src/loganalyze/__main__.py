# ==============================================================================
# Program    : LogAnalyze Package Entry Point (__main__.py)
# Objective  : Allow running package via `python -m loganalyze`.
# Concept    : Module Entry Point
# Why Used   : Standard Python convention for executable packages.
# ==============================================================================

import sys
from loganalyze.main import main

if __name__ == "__main__":
    sys.exit(main())
