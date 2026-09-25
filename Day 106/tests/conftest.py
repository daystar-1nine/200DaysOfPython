"""
Pytest configuration and path setup for Day 106 test suite.
"""

import sys
from pathlib import Path

# Add Day 106 root directory to sys.path
DAY_106_DIR = Path(__file__).resolve().parent.parent
if str(DAY_106_DIR) not in sys.path:
    sys.path.insert(0, str(DAY_106_DIR))
