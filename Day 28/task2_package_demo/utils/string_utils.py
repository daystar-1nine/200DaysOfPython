# ==============================================================================
# Module     : Package String Utilities (Task 2)
# Objective  : String helper functions inside a structured package subfolder.
# Concept    : Package Modules
# Why Used   : Provides string formatting operations as part of the utils package.
# ==============================================================================

def reverse_string(text: str) -> str:
    return text[::-1]

def capitalize_words(text: str) -> str:
    return text.title()
