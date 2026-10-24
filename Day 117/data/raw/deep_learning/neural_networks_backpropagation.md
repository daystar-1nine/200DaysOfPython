# Artificial Neural Networks and the Backpropagation Algorithm

Artificial neural networks compose layers of interconnected artificial neurons computing affine matrix transformations followed by non-linear activation functions: y = f(W*x + b). Multi-layer perceptrons (MLPs) act as universal function approximators capable of modeling non-linear decision boundaries.

The backpropagation algorithm calculates the partial derivatives of the scalar loss with respect to every weight and bias parameter in the network via the chain rule of calculus. Error signals are propagated backwards from the output loss layer through hidden layers to input layers. Modern deep learning frameworks construct dynamic computation graphs during the forward pass to automatically compute and accumulate these gradients during the backward pass.
