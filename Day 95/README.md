<div align="center">

# Advanced YOLO Object Detection & Evaluation Engine

**A professional diagnostic suite to optimize bounding-box regression, validate augmented pipelines, and mathematically evaluate inference latency.**

[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](#)

</div>

---

## 📌 Overview

The **Advanced YOLO Evaluation Engine** moves past simply running inference on sample images and steps into ML Engineering. Today we mathematically define why models fail by dissecting the **Precision-Recall Curve**, exploring multi-scale `Neck` architectures, and pinpointing Localization vs Classification errors using IoU (Intersection over Union) matrices.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 95 / 200).

---

## 🎯 Objectives

- **Dataset Auditing**: Rigorously validating normalized coordinates (`x_center`, `width` within `[0,1]`) before they hit the GPU.
- **Precision vs Recall Thresholds**: Calculating the exact inverse relationship between Confidence Thresholds and False Negatives/Positives.
- **mAP@0.50:0.95 Calculations**: Moving beyond simple single-IoU thresholds to average precision across 10 escalating strictness intervals.
- **Error Analysis**: Creating distinct workflows to identify *Poor Localization* versus *Wrong Classifications* versus *Small Object Failures*.
- **Inference Optimization**: Utilizing `time.perf_counter()` to benchmark actual FPS and Latency metrics.

---

## 🏗️ Technical Architecture

### Component Breakdown
1. **Backbone**: CNN extracting localized spatial features.
2. **Neck**: Feature Pyramids merging high-res spatial grids with low-res semantic grids (Crucial for detecting small objects).
3. **Head**: Generates Box Regression (`L_box`), Classification (`L_class`), and Objectness Probability.

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/yolo_experiments.csv`: *(Metrics reflect CI mock processing outputs).*

| Experiment ID | Model Name | Size | Epochs | mAP@0.50 | mAP@0.50:0.95 | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP001** | YOLO Baseline 20E | 640 | 20 | 0.900 | 0.670 | 222 |
| **EXP002** | YOLO Baseline 40E | 640 | 40 | 0.920 | 0.690 | 222 |
| **EXP003** | YOLO High Conf | 640 | 40 | 0.920 | 0.690 | 222 |
| **EXP004** | YOLO Small Img | 320 | 40 | 0.920 | 0.690 | 222 |
| **EXP005** | YOLO Tuned | 640 | 40 | 0.920 | 0.690 | 222 |

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 95"
   ```
2. **Execute Advanced Benchmark Sweep**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions (Dataset Validations)**:
   ```bash
   python -m pytest tests/
   ```
