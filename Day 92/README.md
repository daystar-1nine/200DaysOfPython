<div align="center">

# ResNet Transfer Learning & Deep Vision Benchmark Engine

**A highly analytical Computer Vision engine designed to compare from-scratch CNNs against ultra-deep ResNet-50 configurations.**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](#)

</div>

---

## 📌 Overview

The **ResNet Transfer Learning Benchmark Engine** moves past simple Transfer Learning and dives deep into architectural optimization. This project specifically evaluates why the **Residual Learning** methodology (`F(x) + x`) utilizing skip connections solves the *Degradation Problem* and the *Vanishing Gradient Problem*.

We pit our Custom CNN architectures against massive `ResNet-50` ImageNet embeddings—testing the exact point at which fine-tuning (at low learning rates like `1e-5`) surpasses frozen feature extraction on the Fashion-MNIST dataset.

This project was built as part of the **200 Days of Python + Data Science** challenge (Day 92 / 200).

---

## 🎯 Objectives

- **Residual Block Architecture**: Leveraging `F(x) + x` skip connections to bypass depth degradation.
- **Bottleneck Block Mechanisms**: Compressing channels using `1x1` Convolutions before computationally expensive `3x3` matrices.
- **Micro-Targeted Fine-Tuning**: Incrementally unfreezing deeper layers (10, 20, 30 layers) to adapt high-level Task-Specific representations without obliterating foundational geometry.
- **Domain Similarity Benchmarks**: Structuring explicit empirical testing to prove/disprove if an ImageNet backbone actually improves performance on 28x28 Grayscale images.

---

## 🏗️ Architecture Matrix

### 1. The Skip Connection Paradigm
```text
          ┌─────── Identity Mapping (x) ────────┐
          │                                     ↓
Input (x) ┴─→ [1x1 Conv] ─→ [3x3 Conv] ─→ [1x1 Conv] ─→ Add (F(x) + x) ─→ ReLU
```

### 2. Fine-Tuning Depths Evaluated
- `Frozen`: 0 trainable parameters in backbone (`1e-3` LR on Head)
- `FT-10`: Deepest 10 layers unfrozen (`1e-5` LR)
- `FT-20`: Deepest 20 layers unfrozen (`1e-5` LR)
- `FT-30`: Deepest 30 layers unfrozen (`1e-5` LR)

---

## 🔬 Benchmark Results

Metrics generated dynamically in `output/benchmark_results.csv`: *(Metrics reflect CI mock processing outputs).*

| Experiment ID | Model Name | Learning Rate | Test Accuracy | Trainable Params |
| :--- | :--- | :--- | :--- | :--- |
| **EXP001** | Dense Baseline | 0.0010 | 0.90 | 20000 |
| **EXP002** | Custom CNN | 0.0010 | 0.90 | 20000 |
| **EXP003** | Frozen ResNet | 0.0010 | 0.90 | 20000 |
| **EXP004** | ResNet FT-10 | 0.00001 | 0.90 | 20000 |
| **EXP005** | ResNet FT-20 | 0.00001 | 0.90 | 20000 |
| **EXP006** | ResNet FT-30 | 0.00001 | 0.90 | 20000 |

---

## 🚀 How to Run

1. **Enter Directory**:
   ```bash
   cd "Day 92"
   ```
2. **Execute Benchmark Sweep**:
   ```bash
   python -m app.main
   ```
3. **Execute 75+ Testing Assertions**:
   ```bash
   python -m pytest tests/
   ```
