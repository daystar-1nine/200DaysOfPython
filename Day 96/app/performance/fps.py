
import time

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.perf_counter()
        
    def update(self):
        self.frame_count += 1
        
    def value(self):
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
