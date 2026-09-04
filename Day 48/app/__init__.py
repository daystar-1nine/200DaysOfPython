"""E-Commerce Backend V4 Package Initialization."""
import os
import sys

pkg_root = os.path.abspath(os.path.dirname(__file__))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

__version__ = "4.0.0"
