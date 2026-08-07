# ==============================================================================
# Program    : Python Project Structure Creator
# Objective  : Programmatically create standard Python project directory layouts.
# Concept    : Modular Folder Structuring & Boilerplate Setup
# Why Used   : Automates creating src/, tests/, utils/, docs/, and config files for new projects.
# ==============================================================================

import os

def create_project_layout(base_path, layout_dict):
    """Recursively creates directory structure and files from a dictionary tree."""
    for name, content in layout_dict.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_project_layout(path, content)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)

def main():
    print("=== PYTHON PROJECT LAYOUT GENERATOR ===")
    
    sample_structure = {
        "src": {
            "__init__.py": "# Package marker\n",
            "main.py": "# Entry point\n"
        },
        "tests": {
            "__init__.py": "",
            "test_sample.py": "def test_pass(): assert True\n"
        },
        "requirements.txt": "requests\n",
        ".gitignore": "venv/\n.env\n__pycache__/\n",
        "README.md": "# Sample Project\n"
    }

    target_dir = os.path.join(os.path.dirname(__file__), "temp_demo_app")
    create_project_layout(target_dir, sample_structure)
    print(f"Successfully generated project layout template at: {target_dir}")

    # Cleanup demo directory tree
    import shutil
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print("Cleaned up temporary project template.")

if __name__ == "__main__":
    main()
