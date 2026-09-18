
import pytest
from app.tracking.centroid_tracker import CentroidTracker
from app.analytics.trajectory import calculate_distance, zone_occupancy

def test_centroid_registration():
    ct = CentroidTracker()
    objects = ct.update([(10, 10, 30, 30)])
    assert len(objects) == 1
    assert 0 in objects
    assert objects[0] == (20, 20)

def test_centroid_tracking_persistence():
    ct = CentroidTracker()
    ct.update([(10, 10, 30, 30)]) # Frame 1
    objects = ct.update([(15, 10, 35, 30)]) # Frame 2, moved slightly
    assert len(objects) == 1
    assert 0 in objects

def test_trajectory_distance():
    pts = [(0, 0), (3, 4)]
    assert calculate_distance(pts) == 5.0

def test_zone_occupancy():
    zones = {'A': (0, 0, 100, 100)}
    assert zone_occupancy((50, 50), zones) == 'A'
    assert zone_occupancy((150, 150), zones) is None
