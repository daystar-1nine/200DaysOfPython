# ==============================================================================
# Program    : Pip Dependency Checker & Requirements Manifest Parser
# Objective  : Parse, validate, and generate requirements.txt files programmatically.
# Concept    : Package Dependency Management
# Why Used   : Parses version specifiers (==, >=) and exports requirements manifests.
# ==============================================================================

import os

# Sample dependency package specifications
SAMPLE_DEPENDENCIES = [
    "requests==2.32.0",
    "flask==3.0.0",
    "python-dotenv==1.0.1",
    "numpy>=2.0.0"
]

def generate_requirements_file(deps, filename="requirements_demo.txt"):
    """Generates a requirements.txt file from list of dependency strings."""
    # What is used : File writing with line splits
    with open(filename, "w", encoding="utf-8") as file:
        for dep in deps:
            file.write(f"{dep}\n")
    print(f"Generated requirements manifest '{filename}' with {len(deps)} dependencies.")

def parse_requirements_file(filename="requirements_demo.txt"):
    """Parses a requirements.txt file and displays package name and version constraints."""
    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        return

    print(f"\n--- Parsing '{filename}' ---")
    with open(filename, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]
        for line in lines:
            print(f"Package Dependency: {line}")

def main():
    print("=== PIP DEPENDENCY MANIFEST DEMO ===")
    demo_file = "temp_requirements.txt"
    generate_requirements_file(SAMPLE_DEPENDENCIES, demo_file)
    parse_requirements_file(demo_file)

    # Cleanup temp file
    if os.path.exists(demo_file):
        os.remove(demo_file)

if __name__ == "__main__":
    main()
