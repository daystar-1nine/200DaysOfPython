# ==============================================================================
# Program    : REST API Explorer Main Driver (main.py)
# Objective  : Entry point driver for running REST API Explorer interactive CLI.
# Concept    : Application Entry Point
# Why Used   : Launches CLI menu loop cleanly.
# ==============================================================================

import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from api_explorer.cli import run_cli_menu

def main() -> int:
    try:
        run_cli_menu()
        return 0
    except KeyboardInterrupt:
        print("\n\nSession cancelled by user. Exiting")
        return 0
    except Exception as e:
        print(f"\nFatal Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
