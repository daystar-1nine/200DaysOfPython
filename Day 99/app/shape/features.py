
import cv2
import numpy as np
import math

def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    # Assume largest contour is the instance
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    
    aspect_ratio = w / float(h) if h > 0 else 0
    circularity = (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0
    
    return {
        'area': area,
        'perimeter': round(perimeter, 2),
        'width': w,
        'height': h,
        'aspect_ratio': round(aspect_ratio, 2),
        'circularity': round(circularity, 2),
        'cx': round(cx, 2),
        'cy': round(cy, 2)
    }
