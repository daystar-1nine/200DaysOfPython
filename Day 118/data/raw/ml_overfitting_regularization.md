# Mitigating Overfitting in Machine Learning Models

Overfitting occurs when a statistical model memorizes noise and idiosyncrasies in the training set rather than learning underlying generalizable patterns, leading to low training loss but high validation loss.

Core techniques to reduce overfitting include:
1. L2 Regularization (Weight Decay / Ridge): Penalizes large weight magnitudes by adding $\frac{\lambda}{2} \sum w_i^2$ to the loss function, smoothing the decision boundary.
2. L1 Regularization (Lasso): Adds $\lambda \sum |w_i|$ to the objective, inducing parameter sparsity for automatic feature selection.
3. Early Stopping: Halts training when validation loss stops improving for a specified patience window, preventing the model from fitting training noise.
4. Data Augmentation: Synthetically expands the dataset via rotations, cropping, or noise injection to improve invariant representations.
5. Dropout: Randomly deactivates neurons during forward passes with probability $p$, preventing complex co-adaptations.