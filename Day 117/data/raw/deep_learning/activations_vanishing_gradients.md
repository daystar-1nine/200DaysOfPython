# Vanishing Gradients and Non-linear Activation Functions

During backpropagation through deep neural networks, gradients can shrink exponentially if activation functions saturate. Classical sigmoid and tanh activations compress inputs into bounded ranges where derivative magnitudes peak below 0.25 and 1.0 respectively; multiplying these fractional derivatives across many layers drives early gradients toward zero, halting learning.

The Rectified Linear Unit (ReLU) activation, defined as max(0, x), avoids vanishing gradients for positive inputs because its derivative is exactly 1.0. However, ReLU can suffer from the 'dying ReLU' phenomenon when neurons receive large negative updates and permanently output zero. Variants like LeakyReLU, Parametric ReLU (PReLU), and Gaussian Error Linear Units (GELU) provide smooth non-zero gradients for negative inputs.
