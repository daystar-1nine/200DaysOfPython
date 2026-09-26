import sys
from pathlib import Path

# Add the Day 107 directory to sys.path so that 'app' can be imported in tests
day_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(day_dir))
