<div align="center">

# Image Segmentation & Mask Analytics Engine

**Extracting exact pixel boundaries utilizing U-Net Encoder-Decoder architectural principles.**

[![Segmentation](https://img.shields.io/badge/Task-Semantic%20Segmentation-9cf.svg)](#)
[![U-Net](https://img.shields.io/badge/Architecture-U--Net-blueviolet.svg)](#)

</div>

---

## 📌 Overview

The **Image Segmentation Engine** leaves rigid bounding boxes behind. Instead of predicting a `4-coordinate vector`, we are predicting a `409,600-coordinate matrix` (for a 640x640 image).

Today we engineered the mathematical foundations for evaluating **Binary Pixel Masks**, implementing the strict matrix calculations for `IoU` and the `Dice Coefficient`, and benchmarking how a baseline CNN compares against the symmetric up-sampling capabilities of a **U-Net**.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 98 / 200).

---

## 🎯 Objectives

- **Pixel Matrix Evaluation**: Calculating the explicit Intersection over Union arrays using `numpy.logical_and()` against flattened ground-truth masks.
- **Dice Coefficients**: Optimizing for the Dice Loss formula (`(2 * TP) / (2TP + FP + FN)`) to combat heavy foreground-background class imbalances.
- **U-Net Principles**: Analyzing how `Skip Connections` transfer high-resolution spatial edges straight from the Encoder to the Decoder before the transposed convolutions blur the boundaries.
- **Threshold Analysis**: Measuring how manipulating the binary cutoff probability (e.g., `0.5` vs `0.8`) scales Precision vs Recall at the pixel level.

---

## 🏗️ Technical Architecture

### The U-Net Segmentation Pipeline
```text
      Input Image (640x640x3)
                   │
         [Encoder (Downsampling)]
                   │
             (Bottleneck)
                   │
         [Decoder (Upsampling)] ◄─── (Skip Connections)
                   │
        Conv2D(1, activation='sigmoid')
                   │
    Predicted Probability Mask (640x640x1)
                   │
         Binary Threshold (>0.5)
                   │
          Final Pixel Mask
```

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/segmentation_experiments.csv`: *(Metrics reflect CI mock processing outputs).*

| Experiment ID | Model | Description | IoU | Dice |
| :--- | :--- | :--- | :--- | :--- |
| **EXP001** | Simple CNN Baseline | Encoder Only | ~0.45 | ~0.50 |
| **EXP002** | U-Net | Encoder + Decoder | ~0.65 | ~0.72 |
| **EXP003** | U-Net + Augmentation | Transforms applied | ~0.75 | ~0.82 |
| **EXP004** | U-Net + Dice Loss | Optimizing boundaries | ~0.82 | ~0.89 |
| **EXP005** | YOLO Segmentation | Instance Extractor | ~0.85 | ~0.91 |

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 98"
   ```
2. **Execute Segmentation Experiments**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions (Mask Math)**:
   ```bash
   python -m pytest tests/
   ```
