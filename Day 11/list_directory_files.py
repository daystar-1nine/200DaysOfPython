# ==============================================================================
# Program    : List Files in Current Directory
# Objective  : Inspect and list files and folders in current directory using os.listdir().
# Concept    : Directory Listing (os.listdir, os.getcwd)
# Why Used   : os.listdir() returns a Python list of file and folder names in target path.
# ==============================================================================

# What is used : import os
import os

# What is used : os.getcwd()
# Why it is used: Gets current working directory path string
# How it works : Queries process environment working directory
current_dir = os.getcwd()
print(f"Current Working Directory:\n{current_dir}\n")

# What is used : os.listdir(current_dir)
# Why it is used: Returns a list of filenames in target directory
items = os.listdir(current_dir)

print(f"Total items in directory: {len(items)}")
print("--- Items List (First 10 shown) ---")

# What is used : enumerate() loop over items slice items[:10]
for idx, item in enumerate(items[:10], start=1):
    item_type = "[DIR]" if os.path.isdir(os.path.join(current_dir, item)) else "[FILE]"
    print(f"{idx}. {item_type:<7} {item}")
