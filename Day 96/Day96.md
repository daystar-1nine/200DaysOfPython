# 🚀 DAY 96 / 200 — REAL-TIME OBJECT DETECTION

## 🧠 Masterclass: Detection vs Tracking

### The Concept:
When a YOLO model predicts a `[Person]` bounding box in Frame 1, and then predicts another `[Person]` bounding box in Frame 2, the detector itself has *no idea* they are the same person.
To build a system that can count how many people walk through a door (Line-Crossing Analytics), we must integrate **Tracking (Object IDs)**.
By measuring the center `(x, y)` coordinate of `ID: 7` in Frame `N-1` and checking if it crossed a spatial boundary in Frame `N`, we transition from static image geometry into Real-Time Spatial Analytics.
