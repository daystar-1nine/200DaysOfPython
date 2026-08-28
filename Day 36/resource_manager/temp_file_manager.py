# ==============================================================================
# Program    : Temporary File Context Managers (temp_file_manager.py)
# Objective  : Provide temporary file creation with guaranteed automatic deletion on exit.
# Concept    : RAII Resource Cleanup Pattern
# Why Used   : Ensures scratch data files are safely deleted when with block exits.
# ==============================================================================

from contextlib import contextmanager
import os
import tempfile

class TemporaryFileManager:
    """Class-based temporary file context manager."""
    def __init__(self, filename: str | None = None, content: str = ""):
        self.filename = filename or os.path.join(tempfile.gettempdir(), f"temp_day36_{os.getpid()}.tmp")
        self.content = content

    def __enter__(self) -> str:
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(self.content)
        print(f"[TEMP] Created temporary file '{self.filename}'")
        return self.filename

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if os.path.exists(self.filename):
            try:
                os.remove(self.filename)
                print(f"[TEMP] Auto-cleaned temporary file '{self.filename}'")
            except OSError as e:
                print(f"[TEMP] Warning: Failed removing file: {e}")
        return False

@contextmanager
def temp_file(content: str = ""):
    """Generator-based temporary file context manager."""
    filename = os.path.join(tempfile.gettempdir(), f"temp_gen_day36_{os.getpid()}.tmp")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass
