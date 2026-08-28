"""LogAnalyze CLI Package Initialization."""
import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from commands import parse_and_execute

__all__ = ["parse_and_execute"]
