
def validate_bbox(x, y, w, h):
    return (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)

def audit_dataset(annotations):
    invalid_count = 0
    for ann in annotations:
        if not validate_bbox(*ann[1:]):
            invalid_count += 1
    return invalid_count
