<div align="center">

# Real-Time YOLO Video Analytics Engine

**Bridging the gap between static image inference and continuous real-time temporal tracking.**

[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-white.svg?logo=opencv&logoColor=red)](#)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](#)

</div>

---

## 📌 Overview

The **Real-Time Video Analytics Engine** explores the mechanical realities of deploying a Computer Vision model in the real world. A webcam transmitting at 30 FPS drops a new spatial matrix onto the CPU every 33 milliseconds. If inference takes 100ms, the system fails.

Today we implemented **Frame Skipping algorithms**, optimized inference resolutions, and engineered a **Line-Crossing Analytics** module that utilizes persistent Object Tracking IDs to determine spatial intent (IN vs OUT counting).

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 96 / 200).

---

## 🎯 Objectives

- **OpenCV Integration**: Managing `cv2.VideoCapture()` pipelines and continuous `while True:` rendering loops.
- **Latency vs Throughput**: Benchmarking pure Inference Latency against total processed Frames Per Second (FPS).
- **Line-Crossing Logic**: Calculating spatial boundary intersections using differential `Y-coordinates` across consecutive frames.
- **Frame Dropping**: Dynamically skipping non-essential frames (`Frame % 2 != 0`) to preserve real-time responsiveness without losing Tracking integrity.

---

## 🏗️ Technical Architecture

### The Temporal Pipeline
```text
      OpenCV Video Stream (30 FPS)
                   │
           [Frame 1] [Frame 2]
                   │
        (Optional Frame Skipping)
                   │
             YOLO Inference
                   │
             Tracker (ID)
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
Spatial Tracking           Line Crossing Logic
(Trajectories)               (IN/OUT Counts)
```

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/video_experiments.csv`: *(Metrics reflect CI mock processing outputs).*

| Experiment ID | Model Name | Image Size | Frame Skip | Processed Frames | IN | OUT | Avg FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP001** | YOLO Baseline | 640 | 1 | 100 | 0 | 5 | ~900 |
| **EXP002** | YOLO Fast | 320 | 1 | 100 | 0 | 5 | ~900 |
| **EXP003** | YOLO Skip 2 | 640 | 2 | 50 | 0 | 3 | ~900 |
| **EXP004** | YOLO Skip 3 | 640 | 3 | 33 | 0 | 2 | ~900 |

*(Note: Dropping frames linearly increases effective FPS latency response, but risks missing sudden line-crossings if the object moves too fast between processed frames).*

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 96"
   ```
2. **Execute Video Pipeline Sweep**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions**:
   ```bash
   python -m pytest tests/
   ```
