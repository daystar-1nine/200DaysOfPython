# Detection Evaluation Metrics: IoU, NMS, and mAP

Intersection over Union (IoU) measures spatial overlap by dividing the area of overlap by the area of union between two bounding boxes: IoU = Area(A cap B) / Area(A cup B). An IoU threshold of 0.5 or 0.7 determines whether a predicted bounding box is considered a True Positive.

Non-Maximum Suppression (NMS) eliminates duplicate candidate boxes for the same physical object. NMS greedily selects the highest-scoring candidate box and discards all neighboring boxes whose IoU overlap exceeds a predefined threshold. Mean Average Precision (mAP) evaluates overall detector performance by averaging precision across recall levels and semantic categories (e.g. mAP@0.5 and mAP@0.5:0.95).
