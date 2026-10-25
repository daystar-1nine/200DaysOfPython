# Backpropagation and Reverse-Mode Automatic Differentiation

Backpropagation computes partial derivatives of the scalar objective loss with respect to all network parameters using the multivariate chain rule of calculus.
During the forward pass, intermediate activations are computed and cached in the computational graph.
During the backward pass, adjoint sensitivities flow in reverse order from the output loss back to initial input weights.

Modern deep learning frameworks (such as PyTorch and JAX) implement reverse-mode automatic differentiation (Autograd), which calculates gradients in time proportional to a single forward pass.