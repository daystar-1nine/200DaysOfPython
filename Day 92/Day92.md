# 🚀 DAY 92 / 200 — RESNET & RESIDUAL LEARNING

## 🧠 Masterclass: Beating the Vanishing Gradient

### The Concept:
Why can ResNet be 152 layers deep, but standard CNNs tap out at ~20 layers? The answer is `H(x) = F(x) + x`. Skip connections allow gradients to skip over processing blocks unchanged during Backpropagation, totally bypassing the Vanishing Gradient problem. By utilizing 1x1 convolutions in Bottleneck Blocks, ResNet compresses parameter counts while allowing insane depth!
