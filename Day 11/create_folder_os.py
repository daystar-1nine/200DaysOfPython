# ==============================================================================
# Program    : Create a Folder Using OS Module
# Objective  : Safely create nested folder paths using os.makedirs().
# Concept    : Operating System File System Interaction (os module)
# Why Used   : os.makedirs(..., exist_ok=True) creates directories safely without crashing if they exist.
# ==============================================================================

# What is used : import os
# Why it is used: Provides platform-independent operating system interface
import os

target_folder = "sample_output_dir"

# What is used : os.makedirs(path, exist_ok=True)
# Why it is used: Creates directory tree path safely; exist_ok=True suppresses error if folder exists
# How it works : System call to OS file system daemon to allocate directory entry
try:
    os.makedirs(target_folder, exist_ok=True)
    print(f"Folder '{target_folder}' verified/created successfully!")
    print(f"Absolute Path: {os.path.abspath(target_folder)}")
except Exception as e:
    print(f"OS Error creating folder: {e}")
