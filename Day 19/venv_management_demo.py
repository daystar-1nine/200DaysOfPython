# ==============================================================================
# Program    : Virtual Environment Inspection Utility
# Objective  : Inspect active Python environment and verify virtualenv status.
# Concept    : Environment Isolation Check via sys.prefix & sys.base_prefix
# Why Used   : Determines if execution is isolated inside a venv environment.
# ==============================================================================

import os
import sys

def check_virtual_environment():
    # What is used : sys.prefix vs sys.base_prefix comparison
    # Why it is used: In virtual environments, sys.prefix points to venv folder while base_prefix points to system Python
    in_venv = sys.prefix != sys.base_prefix

    print("=== PYTHON VIRTUAL ENVIRONMENT INSPECTION ===")
    print(f"Running in Virtual Environment : {in_venv}")
    print(f"Active Environment Path (sys.prefix): {sys.prefix}")
    print(f"System Base Python Path (base_prefix): {sys.base_prefix}")
    print(f"Current Executable Path             : {sys.executable}")
    print(f"Python Version                      : {sys.version.split()[0]}")
    
    if in_venv:
        print("\n[STATUS] Environment is ISOLATED! Dependencies installed here won't affect global Python.")
    else:
        print("\n[STATUS] Running on System/Global Python environment.")

def main():
    check_virtual_environment()

if __name__ == "__main__":
    main()
