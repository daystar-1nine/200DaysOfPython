# ==============================================================================
# Program    : Create a Folder Using OS Module
# Objective  : Safely create directory folders using os.makedirs().
# Concept    : Operating System File System Operations (os module)
# Why Used   : os.makedirs(..., exist_ok=True) creates directory trees without throwing errors if folder exists.
# ==============================================================================

# What is used : import os
# Why it is used: Provides platform-independent OS file system interface
import os

target_folder = "sample_output_dir"

# What is used : os.makedirs(path, exist_ok=True)
# Why it is used: Creates directory folder safely; exist_ok=True prevents crash if folder already exists
# How it works : Makes a low-level OS system call to create folder entry in file table
try:
    os.makedirs(target_folder, exist_ok=True)
    print(f"Folder '{target_folder}' verified/created successfully!")

    # What is used : os.path.abspath(path)
    # Why it is used: Converts relative folder name into full absolute system path
    print(f"Absolute Path: {os.path.abspath(target_folder)}")

except Exception as e:
    print(f"OS Error creating folder: {e}")
