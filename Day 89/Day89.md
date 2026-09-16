# 🚀 DAY 89 / 200 — CONVOLUTIONAL NEURAL NETWORKS

## 🧠 Masterclass: The Mathematics of Computer Vision

### Convolution Mechanics
1. **Kernels and Feature Maps:** A CNN slides a small filter grid (kernel) across spatial axes of a multi-dimensional array (Tensor). The dot products generated establish a *feature map*, effectively detecting spatial relations rather than blind tabular inputs.
2. **Translation Invariance via Parameter Sharing:** Because the exact same kernel weights map across the entire image dimension, the network can recognize a feature (like an edge or corner) regardless of where it lies locally.
3. **Stride & Pooling:** Max Pooling destructively extracts the highest dimensional values in local patches to reduce overall parameters downstream, allowing CNNs to computationally scale deeply without immediately running out of video RAM.

> [!IMPORTANT]
> Dense nets require unique weights for every input vector point. CNNs completely avoid this via parameter sharing!
