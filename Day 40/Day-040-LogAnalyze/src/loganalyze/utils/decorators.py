# ==============================================================================
# Program    : Performance Timer Decorator (decorators.py)
# Objective  : Measure analysis function execution duration (Day 33 requirement).
# Concept    : Function Decorators & Metadata Preservation (@wraps)
# Why Used   : Profiles analysis speed and attaches last execution time metadata.
# ==============================================================================

from functools import wraps
import time
from typing import Any, Callable

def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator profiling execution duration of analysis functions in seconds."""
    # What is used : @wraps(func)
    # Why it is used: Preserves decorated function's __name__ and __doc__ metadata
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_t = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_t
        wrapper.last_execution_time = elapsed  # type: ignore
        return result
    wrapper.last_execution_time = 0.0  # type: ignore
    return wrapper
