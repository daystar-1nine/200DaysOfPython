
import math

def calculate_distance(points):
    if len(points) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(points)):
        dist += math.dist(points[i-1], points[i])
    return dist

def zone_occupancy(centroid, zones):
    # zones: dict of {'zoneA': (x1, y1, x2, y2)}
    for zone_name, (x1, y1, x2, y2) in zones.items():
        if x1 <= centroid[0] <= x2 and y1 <= centroid[1] <= y2:
            return zone_name
    return None
