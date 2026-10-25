# Gradient Descent Optimization and Learning Rate Tuning

Gradient descent is a first-order iterative optimization algorithm used to minimize the cost function $J(\theta)$ by updating model parameters in the direction of the negative gradient.
The fundamental parameter update formula is:
$$\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)$$
where $\eta$ is the learning rate.

The learning rate $\eta$ is the single most critical hyperparameter in optimization:
1. Too small $\eta$: Convergence is excruciatingly slow, and optimization risks being trapped in shallow sub-optimal local minima.
2. Too large $\eta$: Updates overshoot the minimum, causing loss oscillation or numerical divergence ($NaN$).

Adaptive optimizers such as Adam, RMSprop, and AdaGrad scale individual coordinate learning rates using historical first and second moment statistics of the gradients.
Learning rate schedules (such as cosine annealing with warmup) systematically decay $\eta$ across epochs to stabilize late-stage convergence.