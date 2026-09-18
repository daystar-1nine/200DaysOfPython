<div align="center">

# Instance Segmentation & Object Shape Analytics Engine

**Isolating unique object matrices and executing morphological spatial measurements.**

[![Instance Segmentation](https://img.shields.io/badge/Task-Instance%20Segmentation-ff69b4.svg)](#)
[![OpenCV Analytics](https://img.shields.io/badge/Morphology-OpenCV%20Contours-orange.svg)](#)

</div>

---

## 📌 Overview

The **Instance Analytics Engine** bridges the gap between Deep Learning and traditional Computer Vision. Once a model like `YOLOv8-Seg` or `Mask R-CNN` outputs a probability matrix, we convert that matrix into a binary contour. 

Today we engineered the mathematical foundations for evaluating **Instance Shape**. We implemented the formulas for `Circularity`, `Aspect Ratio`, and `Centroid Moments`, and utilized morphological `Opening` and `Closing` kernels to eliminate prediction noise.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 99 / 200). One day from the 50% milestone!

---

## 🎯 Objectives

- **Instance Isolation**: Transitioning from class-level semantic maps to unique object-ID boolean arrays.
- **Morphological Cleaning**: Applying Elliptical Structuring Kernels via `cv2.morphologyEx` to execute `Opening` (Erosion followed by Dilation) to strip stray prediction pixels without destroying the core geometry.
- **Shape Feature Extraction**: Utilizing `cv2.findContours` and `cv2.moments` to dynamically calculate the physical `x,y` center of mass, the pixel Area, and the perimeter Arc Length.
- **Circularity Physics**: Implementing the mathematical roundness ratio to determine object deformities.

---

## 🏗️ Technical Architecture

### The Instance Analytics Pipeline
```text
  Instance Probability Mask (from YOLO/Mask R-CNN)
                   │
         [Binary Thresholding]
                   │
         [Morphological Opening] (Noise reduction)
                   │
           cv2.findContours()
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
  Area         Centroid      Circularity
```

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/instance_metrics.csv`: *(Metrics reflect CI mock processing outputs).*

| id | class | confidence | area | perimeter | width | height | aspect_ratio | circularity | cx | cy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | apple | 0.95 | 693.0 | 92.28 | 31 | 31 | 1.00 | 1.02 | 30.00 | 30.00 |
| **2** | bottle | 0.88 | 570.0 | 98.00 | 21 | 31 | 0.68 | 0.74 | 70.00 | 75.00 |

*(Notice the mathematical verification: The circular `apple` mock generates an Aspect Ratio of 1.0 and a Circularity of ~1.0, while the rectangular `bottle` mock registers a lower circularity of 0.74).*

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 99"
   ```
2. **Execute Instance Shape Extraction**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions (Morphology/Contours)**:
   ```bash
   python -m pytest tests/
   ```
