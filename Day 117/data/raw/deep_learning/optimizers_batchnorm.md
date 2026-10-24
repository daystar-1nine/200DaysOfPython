# Adaptive Optimization Algorithms and Batch Normalization

Modern deep neural networks rely on adaptive learning rate optimizers. The Adam (Adaptive Moment Estimation) optimizer tracks exponential moving averages of both past gradients (first moment, momentum) and squared gradients (second moment, uncentered variance), providing adaptive per-parameter learning rate scaling and robust convergence across sparse gradients.

Batch Normalization (BatchNorm) normalizes activations across the mini-batch dimension during training: x_hat = (x - mean) / sqrt(var + eps), followed by learnable scale (gamma) and shift (beta) transformations. BatchNorm stabilizes internal covariate shift, smooths the optimization loss landscape, and allows significantly higher learning rates while acting as a mild regularizer.
