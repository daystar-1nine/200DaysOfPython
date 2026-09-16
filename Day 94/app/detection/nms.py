
from app.detection.iou import calculate_iou

def nms(boxes, scores, iou_threshold=0.5):
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices.pop(0)
        keep.append(current)

        remaining = []
        for index in indices:
            iou = calculate_iou(boxes[current], boxes[index])
            if iou < iou_threshold:
                remaining.append(index)

        indices = remaining

    return keep
