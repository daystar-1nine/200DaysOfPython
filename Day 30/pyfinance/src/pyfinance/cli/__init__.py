"""PyFinance CLI Presentation Package."""
import os
import sys

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from pyfinance.cli.commands import setup_cli_parser, execute_cli_command

__all__ = ["setup_cli_parser", "execute_cli_command"]
