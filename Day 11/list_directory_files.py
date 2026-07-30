# ==============================================================================
# Program    : List Files in Current Directory
# Objective  : Inspect and list files and folders in working directory using os.listdir().
# Concept    : Directory Traversal (os.listdir, os.getcwd, os.path.isdir)
# Why Used   : os.listdir() returns a Python list containing names of all entries in target directory.
# ==============================================================================

# What is used : import os
import os

# What is used : os.getcwd()
# Why it is used: Returns current working directory path string
# How it works : Queries process environment for active directory path
current_dir = os.getcwd()
print(f"\nCurrent Working Directory:\n{current_dir}\n")

# What is used : os.listdir(current_dir)
# Why it is used: Retrieves a Python list of all file and directory entries in target path
items = os.listdir(current_dir)

print(f"Total items in directory: {len(items)}")
print("--- Items List (First 10 shown) ---")

# What is used : enumerate() loop over list slice items[:10]
# Why it is used: Provides 1-based index numbers and limits display to first 10 items
for idx, item in enumerate(items[:10], start=1):
    # What is used : os.path.isdir(full_path)
    # Why it is used: Determines whether entry is a directory folder or a file
    full_item_path = os.path.join(current_dir, item)
    item_type = "[DIR]" if os.path.isdir(full_item_path) else "[FILE]"
    print(f"{idx}. {item_type:<7} {item}")
