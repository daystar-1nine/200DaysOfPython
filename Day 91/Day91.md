# 🚀 DAY 91 / 200 — TRANSFER LEARNING & PRETRAINED CNNs

## 🧠 Masterclass: Standing on the Shoulders of Giants

### The Architecture:
We freeze an entire pre-trained `MobileNetV2` trained on `ImageNet` (millions of natural photos) to extract shapes, edges, and textures directly from our Fashion-MNIST dataset. We swap out the end-layer classification flat mapping for `GlobalAveragePooling2D` which dramatically slashes parameters.
