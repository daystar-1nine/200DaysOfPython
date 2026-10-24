# Gradient Descent Optimization Algorithms in Machine Learning

Gradient descent minimizes empirical risk loss functions by iteratively updating parameter weights in the direction opposite to the gradient of the loss with respect to parameters: w = w - lr * grad. Batch Gradient Descent computes the exact gradient over the entire training corpus, ensuring smooth monotonic convergence but incurring heavy computational and memory costs for large datasets.

Stochastic Gradient Descent (SGD) updates parameters using a single random sample per iteration, introducing high stochastic variance that aids in escaping shallow local minima but leads to noisy convergence paths. Mini-batch Gradient Descent balances both paradigms by computing gradients over small subsets (e.g. 32 to 256 samples), maximizing hardware GPU parallelism while providing robust gradient estimates.
