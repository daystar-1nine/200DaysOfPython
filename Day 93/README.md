<div align="center">

# Efficient CNN & Edge Vision Benchmark Engine

**A targeted benchmarking suite to analyze CNN accuracy tradeoffs against Parameter Counts, Inference Latency, and Memory Footprint.**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](#)
[![Edge AI](https://img.shields.io/badge/Edge-AI-8A2BE2.svg)](#)

</div>

---

## 📌 Overview

The **Efficient CNN Benchmark Engine** drops the paradigm of purely maximizing model accuracy and addresses the engineering constraints of deploying intelligence to resource-constrained environments (Mobile, Embedded Devices, Edge cameras).

We construct a multi-model validation pipeline comparing `ResNet50`—a massive, deep standard convolution network—against `MobileNetV2`, which achieves similar representational capabilities utilizing **Depthwise Separable Convolutions** and **Inverted Residuals** at a fraction of the parameter count and inference cost.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 93 / 200).

---

## 🎯 Objectives

- **Depthwise Separable Convolutions**: Separating Spatial Filtering and Channel Mixing to dramatically reduce parameters (`K^2 * C_in + C_in * C_out`).
- **MobileNetV2 Bottlenecks**: Testing Linear Bottlenecks and Inverted Residual blocks (`Narrow -> Wide -> Narrow`).
- **Edge Inference Timing**: Standardizing explicit `time.perf_counter()` calculations per batch slice.
- **Dimensionality Multipliers**: Identifying computational scaling (Resolution/Width).
- **Efficiency Metric Matrix**: Correlating `Accuracy vs Parameters` and `Accuracy vs Latency`.

---

## 🏗️ Technical Architecture

### 1. Depthwise Convolution Math
Traditional Parameter Count: `9 * 32 * 64 = 18,432`
Depthwise Separable Count: `(9 * 32) + (32 * 64) = 2,336` (nearly an **8x** reduction!)

### 2. Inverted Residual Flow
`Low-Dim Input` -> `1x1 Expansion (Non-linear)` -> `Depthwise Conv` -> `1x1 Projection (Linear)` -> `Low-Dim Output`

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/benchmark_results.csv`: *(Metrics reflect CI mock processing outputs).*

| Experiment ID | Model Name | Test Accuracy | Train Time | Inference Time (ms/img) | Params | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP001** | Dense Baseline | 0.90 | 0.02s | ~0.5 ms | 394,000 | 1.5 MB |
| **EXP002** | Custom CNN | 0.90 | 0.01s | ~0.5 ms | 2,100,000 | 8.0 MB |
| **EXP003** | ResNet50 | 0.90 | 0.01s | ~0.5 ms | 23,000,000 | 92.0 MB |
| **EXP004** | MobileNetV2 | 0.90 | 0.01s | ~0.5 ms | 2,200,000 | 8.8 MB |

*(Observe ResNet's parameter explosion vs MobileNet's compression!)*

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 93"
   ```
2. **Execute Benchmark Sweep**:
   ```bash
   python -m app.main
   ```
3. **Execute Testing Assertions**:
   ```bash
   python -m pytest tests/
   ```
