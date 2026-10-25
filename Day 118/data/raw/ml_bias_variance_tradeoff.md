# The Bias-Variance Tradeoff in Supervised Learning

The expected generalization error of a machine learning model decomposes into three distinct components: Bias error, Variance error, and Irreducible error ($\sigma^2$).
Bias represents error stemming from erroneous simplifying assumptions in the learning algorithm; high bias leads to underfitting.
Variance measures sensitivity to small fluctuations in the training dataset; high variance leads to overfitting.

Ensemble methods directly address this tradeoff:
Bagging (Bootstrap Aggregating, such as Random Forests) reduces variance without increasing bias by averaging predictions from multiple de-correlated base estimators.
Boosting (such as XGBoost or LightGBM) sequentially fits weak learners to residual errors, primarily reducing bias.