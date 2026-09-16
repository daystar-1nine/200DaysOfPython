<div align="center">

# Object Detection Benchmark & YOLO Inference Engine

**The transition from classifying images to localizing multiple bounding-box objects simultaneously in real-time.**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](#)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00FFFF.svg)](#)

</div>

---

## 📌 Overview

The **Object Detection Engine** handles the foundational mathematics necessary to pinpoint multiple subjects within a single matrix. Instead of simply asking *"What is this image?"*, we build a pipeline that asks *"What objects are present, where are they explicitly located, and how confident are we?"*

We built an inference pipeline wrapping **YOLO** (You Only Look Once), an incredibly fast one-stage detector. To truly understand the architecture, we engineered the core filtering algorithms **Intersection over Union (IoU)** and **Non-Maximum Suppression (NMS)** entirely from scratch.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 94 / 200).

---

## 🎯 Objectives

- **Bounding Box Math**: Reformatting coordinates (`x_center, y_center, width, height`) to (`x_min, y_min, x_max, y_max`).
- **Intersection over Union (IoU)**: Manually calculating `Area of Overlap / Area of Union` to validate prediction geometry against Ground Truths.
- **Non-Maximum Suppression (NMS)**: Eliminating clustered duplicate bounding boxes prioritizing Confidence Thresholds.
- **Precision / Recall / mAP**: Evaluating models using Mean Average Precision at strict overlap metrics (mAP@0.50).

---

## 🏗️ Technical Architecture

### The One-Stage Pipeline
```text
                 IMAGE
                   │
                   ▼
          YOLO Feature Extractor
                   │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Object 1    Object 2    Object 3
             │           │           │
          (Box + Confidence + Class Probability)
                   │
                   ▼
       Non-Maximum Suppression (NMS Scrubbing)
                   │
                   ▼
            FINAL PREDICTIONS
```

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/detection_experiments.csv`: *(Metrics reflect CI mock processing outputs).*

| Experiment ID | Model Name | Conf Thresh | Image Size | mAP@0.50 | Inference (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP001** | YOLO Baseline | 0.50 | 640 | 0.88 | ~0.5 ms |
| **EXP002** | YOLO Low Conf | 0.25 | 640 | 0.88 | ~0.5 ms |
| **EXP003** | YOLO High Conf | 0.75 | 640 | 0.88 | ~0.5 ms |
| **EXP004** | YOLO Small Img | 0.50 | 320 | 0.88 | ~0.5 ms |

*(Lower confidence triggers higher Recall but lower Precision; smaller image sizes speed up inference but drop AP on small objects).*

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 94"
   ```
2. **Execute Detection Sweep**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions (IoU & NMS Unit Tests)**:
   ```bash
   python -m pytest tests/
   ```
