# Residual Networks (ResNets) and Identity Skip Connections

Prior to Residual Networks (ResNets), stacking dozens of convolutional layers caused degradation in training accuracy due to optimization difficulties rather than overfitting. ResNets resolve this degradation by introducing identity skip connections (shortcut connections) where input x bypasses weight layers: F(x) + x.

These residual paths allow gradients to flow directly backwards through the network without attenuation, enabling stable gradient backpropagation in architectures exceeding 100 or 1000 layers. Bottleneck residual blocks use 1x1 convolutions before and after 3x3 convolutions to reduce and restore feature dimensions, minimizing computational cost while maximizing representational capacity.
