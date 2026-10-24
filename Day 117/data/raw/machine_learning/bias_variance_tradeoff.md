# The Bias-Variance Tradeoff in Supervised Learning

The bias-variance tradeoff governs generalization error in supervised learning. Bias measures the error originating from erroneous assumptions or over-simplifications in the model learning algorithm, leading to underfitting where the model fails to capture underlying data patterns. Variance measures the model's sensitivity to small fluctuations in the training dataset, leading to overfitting where the model memorizes noise rather than signal.

The total expected generalization error decomposes into Bias^2 + Variance + Irreducible Error. Highly flexible models (such as deep decision trees or unregularized neural networks) typically exhibit low bias but high variance, whereas rigid models (such as linear regression) exhibit high bias and low variance. Cross-validation and regularization help locate the optimal tradeoff point.
