# 🚀 DAY 93 / 200 — MOBILENET, EFFICIENT CNNs & EDGE AI

## 🧠 Masterclass: Trading Off Computational Weights

### The Concept:
A Convolution mathematically performs Spatial Filtering + Channel Mixing in the same operation.
`Depthwise Separable Convolutions` (used in MobileNet) break this down:
1. `Depthwise`: Spatially filter each input channel independently.
2. `Pointwise`: 1x1 Convolutions mapping across all filters.

The result? ResNet accuracy with practically 1/10th the computational cost. Extremely critical for Edge AI deployments!
