# ==============================================================================
# Module     : Reusable Project Core Logic
# Objective  : Core business logic module for python starter template.
# Concept    : Modular Source Code Separation (src/ layer)
# Why Used   : Keeps business logic decoupled from CLI and entry point wrappers.
# ==============================================================================

class CoreEngine:
    def __init__(self, name="Python App"):
        self.name = name

    def get_status(self):
        return f"[ENGINE ONLINE] System '{self.name}' is operational!"

def calculate_square(x):
    """Utility function returning square of input x."""
    return x * x
