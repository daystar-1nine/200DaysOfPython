# 🚀 DAY 88 / 200 — PRACTICAL NEURAL NETWORK TRAINING

## 🧠 Masterclass: Optimizing Deep Learning

### Optimization Mechanics
1. **SGD vs Adam:** Stochastic Gradient Descent provides a reliable mechanism by updating weights against a gradient subset. Adam utilizes Adaptive Moment Estimation, internally tuning learning rates across individual parameters.
2. **Learning Rate Scheduling:** The `ReduceLROnPlateau` algorithm strategically steps down learning rates whenever validation loss stalls, preventing divergence while settling into global minima.

### Regularization
3. **Dropout:** A mechanism that randomly nullifies a fraction of dense activations during training, effectively tearing down parameter co-dependence and preventing extreme overfitting.
4. **L2 Penalty:** Discourages exploding weights by punishing the loss function continuously proportional to the squared weight magnitude.

> [!TIP]
> Use `EarlyStopping` paired directly with a `ModelCheckpoint` to lock in the optimal weights before extreme overfitting kicks in.
