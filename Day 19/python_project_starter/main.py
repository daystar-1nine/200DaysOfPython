# ==============================================================================
# Program    : Main Entry Point (Starter Template)
# Objective  : Launch application using src/ package components.
# Concept    : Application Entry Point Architecture
# Why Used   : Connects package modules and initiates application execution.
# ==============================================================================

import os
import sys

# Ensure src module is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core import CoreEngine, calculate_square

def main():
    print("==========================================================")
    print("            PYTHON PROJECT STARTER TEMPLATE               ")
    print("==========================================================")
    
    engine = CoreEngine("Day 19 Professional Starter")
    print(engine.get_status())
    
    val = 12
    sq = calculate_square(val)
    print(f"Sample Core Engine Operation: Square of {val} = {sq}")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
