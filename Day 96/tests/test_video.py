
import pytest
import time
from app.analytics.line_counter import has_crossed_line
from app.performance.fps import FPSCounter

def test_has_crossed_line_down():
    assert has_crossed_line((100, 290), (100, 310), 300) == 1

def test_has_crossed_line_up():
    assert has_crossed_line((100, 310), (100, 290), 300) == -1

def test_has_crossed_line_no_cross():
    assert has_crossed_line((100, 280), (100, 290), 300) == 0

def test_fps_counter():
    fps = FPSCounter()
    fps.update()
    fps.update()
    time.sleep(0.01)
    assert fps.value() > 0
