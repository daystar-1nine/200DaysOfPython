# Object Detection: Two-Stage vs One-Stage Detectors

Object detection identifies spatial bounding boxes and semantic class labels for multiple objects in an image. Two-stage detectors (such as Faster R-CNN) first generate sparse candidate regions of interest using a Region Proposal Network (RPN) and subsequently classify and refine bounding box coordinates using RoI pooling or RoIAlign.

One-stage detectors (such as YOLO - You Only Look Once, and SSD) frame detection as a single unified regression problem, predicting bounding boxes and class probabilities directly from dense feature map grids in a single forward pass. While two-stage detectors traditionally achieved higher localization accuracy, modern one-stage detectors achieve comparable accuracy at real-time inference speeds (exceeding 60 FPS).
