# ==============================================================================
# Program    : Performance Timer Context Managers (timer_manager.py)
# Objective  : Measure execution duration of code blocks using context managers.
# Concept    : Code Block Performance Profiling
# Why Used   : Tracks execution time of block statements safely.
# ==============================================================================

from contextlib import contextmanager
import time

class TimerManager:
    """Class-based execution timer context manager."""
    def __init__(self, label: str = "Block"):
        self.label = label
        self.elapsed = 0.0
        self._start_time = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed = time.perf_counter() - self._start_time
        print(f"[TIMER] '{self.label}' completed in {self.elapsed:.6f} seconds.")
        return False

@contextmanager
def execution_timer(label: str = "Block"):
    """Generator-based execution timer context manager."""
    start_t = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_t
        print(f"[TIMER] '{label}' completed in {elapsed:.6f} seconds.")
