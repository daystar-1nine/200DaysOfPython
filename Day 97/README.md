<div align="center">

# Multi-Object Tracking & Trajectory Analytics Engine

**Solving the Data Association problem to build resilient, identity-preserving Multi-Object Tracking pipelines.**

[![Tracking](https://img.shields.io/badge/MOT-Centroid%20%7C%20SORT-FF9900.svg)](#)
[![Data Association](https://img.shields.io/badge/Hungarian-Algorithm-green.svg)](#)

</div>

---

## 📌 Overview

The **Multi-Object Tracking (MOT) Engine** moves past simply plotting bounding boxes and tackles the core problem of visual continuity: **Identity**. We mathematically answer the question: *Is the person I see in Frame N the exact same person I saw in Frame N-1?*

Today we implemented the foundational logic for **Centroid Tracking** and designed the architectural structures that power advanced MOT frameworks like **SORT** (Simple Online and Realtime Tracking) and **ByteTrack**.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 97 / 200).

---

## 🎯 Objectives

- **Centroid Distance Matching**: Calculating Euclidean distance costs between known Track centers and incoming Detection centers.
- **Track Lifecycles**: Implementing robust `max_disappeared` buffering to prevent instantaneous Identity loss when YOLO misses a frame due to occlusion.
- **Trajectory Analytics**: Aggregating historical tracked coordinates to output total Euclidean distances traveled and active **Zone Occupancy**.
- **Data Association**: Structuring the Pipeline to accept Hungarian Algorithm assignments and Kalman Filter state predictions.

---

## 🏗️ Technical Architecture

### The MOT Data Association Pipeline
```text
           [Active Tracks (Frame N-1)]
                       │
             (Kalman Filter Predict)
                       │
          ┌────────────▼────────────┐
          │      Cost Matrix        │ 
          │  (Distance / IoU Cost)  │
          └────────────┬────────────┘
                       │
            [New Detections (Frame N)]
                       │
             (Hungarian Algorithm)
                       │
          ┌────────────┴────────────┐
          ▼            ▼            ▼
   Update Track    Delete Track  Create Track
```

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/tracking_experiments.csv`: *(Metrics reflect CI mock processing outputs).*

| Tracker | Unique Identities | Avg Distance (px) | ID Switches | Tracking Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Centroid Tracker Baseline** | 1 | 42.43 | 0 | ~0.02 |
| **SORT (Simulated)** | 1 | 42.43 | 0 | ~0.02 |
| **Deep SORT (Simulated)** | 1 | 42.43 | 0 | ~0.02 |
| **ByteTrack (Simulated)** | 1 | 42.43 | 0 | ~0.02 |

*(Note: The Baseline Centroid Tracker successfully bridged a simulated missed detection (Frame 4) by retaining the `max_disappeared` buffer state, preserving the singular identity).*

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 97"
   ```
2. **Execute Tracking Trajectory Sweep**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions (Centroid/Zone Matching)**:
   ```bash
   python -m pytest tests/
   ```
