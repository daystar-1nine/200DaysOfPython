# 🚀 DAY 97 / 200 — MULTI-OBJECT TRACKING

## 🧠 Masterclass: The Data Association Problem

### The Concept:
Object Detection gives you disconnected spatial bounding boxes (`BBox 1`, `BBox 2`) in Frame 1, and new ones in Frame 2. 
If we want to know *which* Frame 1 BBox corresponds to *which* Frame 2 BBox, we must solve a **Cost Matrix Assignment Problem**.
We can use Centroid Distance, IoU overlap, or Deep Visual Features to calculate a "Cost" between all previous Tracks and all new Detections.
We then use the **Hungarian Algorithm** to mathematically pair them up in a way that minimizes the total global cost. That is the essence of **Data Association**.
