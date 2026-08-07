# ==============================================================================
# Module     : Unit Tests for Core Engine
# Objective  : Automated test suite for testing src/core.py logic.
# Concept    : Automated Testing Architecture (tests/ layer)
# Why Used   : Ensures code correctness before shipping releases.
# ==============================================================================

import os
import sys

# Ensure src module is importable
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.core import CoreEngine, calculate_square

def test_engine_status():
    engine = CoreEngine("TestApp")
    assert "TestApp" in engine.get_status()

def test_square_calculation():
    assert calculate_square(5) == 25
    assert calculate_square(0) == 0

if __name__ == "__main__":
    test_engine_status()
    test_square_calculation()
    print("All unit tests passed successfully! [PASSED]")
