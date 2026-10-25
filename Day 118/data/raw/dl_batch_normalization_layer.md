# Batch Normalization Principles and Covariate Shift

Batch Normalization (BatchNorm) stabilizes deep neural network training by normalizing layer activations across mini-batches during forward propagation.
Given mini-batch activations $x$, BatchNorm calculates batch mean $\mu_B$ and batch variance $\sigma_B^2$, normalizes the inputs, and applies learnable scale $(\gamma)$ and shift $(\beta)$ parameters:
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

BatchNorm smooths the optimization landscape, enables significantly higher learning rates, reduces sensitivity to parameter initialization, and acts as a mild regularizer.