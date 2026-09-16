# 🚀 DAY 94 / 200 — OBJECT DETECTION & YOLO

## 🧠 Masterclass: You Only Look Once

### The Concept:
Image Classification returns `[Cat]`. Object Localization returns `[Cat, (x,y,w,h)]`. Object Detection returns `[Cat, (x,y,w,h)], [Dog, (x,y,w,h)]...`

YOLO bypasses Region Proposals (like R-CNN) and passes the image through the CNN exactly *once*. The network output grid predicts Bounding Boxes, Confidence Scores, and Class Probabilities simultaneously. Then, Non-Maximum Suppression (NMS) scrubs out duplicate predictions heavily overlapping the same object based on the Intersection over Union (IoU) ratio.
