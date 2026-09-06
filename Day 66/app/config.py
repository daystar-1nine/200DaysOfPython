"""
Configuration settings for Day 66 Probability Simulation Engine.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Simulation Parameters
RANDOM_SEED = 2026
DEFAULT_COIN_FLIPS = 50_000
DEFAULT_DICE_ROLLS = 100_000
DEFAULT_TWO_DICE_ROLLS = 200_000
DEFAULT_CARD_DRAWS = 100_000
DEFAULT_CONVERSION_VISITORS = 50_000
