# ==============================================================================
# Program    : LogAnalyze Main CLI Driver (main.py)
# Objective  : Composition root orchestrating CLI commands execution.
# Concept    : Composition Root & Application Entry Point
# Why Used   : Handles CLI argument dispatching and error boundary catching.
# ==============================================================================

import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from loganalyze.cli.commands import parse_and_execute
from loganalyze.exceptions import LogAnalyzeError

def main(args: list[str] | None = None) -> int:
    """Main CLI execution driver."""
    try:
        parse_and_execute(args)
        return 0
    except LogAnalyzeError as err:
        print(f"\n❌ Error [{err.code}]: {err.message}", file=sys.stderr)
        if err.__cause__:
            print(f"   Root Cause: {err.__cause__}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n❌ Unexpected System Error: {err}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
