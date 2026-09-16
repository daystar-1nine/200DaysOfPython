# Challenge 1: Euclidean Distance
def euclidean_distance(a, b):
    import math
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
