"""
Pytest configuration for Day 110 test suite.
Configures python path for seamless imports across scratch and app modules.
"""
import sys
from pathlib import Path

day_dir = Path(__file__).resolve().parent.parent
scratch_dir = day_dir / "scratch"

for p in [str(day_dir), str(scratch_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)
