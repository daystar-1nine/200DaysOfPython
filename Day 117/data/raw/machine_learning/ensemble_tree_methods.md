# Ensemble Learning: Bagging, Random Forests, and Gradient Boosting

Ensemble methods combine multiple base estimators to produce superior predictive accuracy and robustness. Bagging (Bootstrap Aggregating) trains diverse base models in parallel on independent bootstrap samples drawn with replacement from the training set, reducing variance without increasing bias. Random Forests extend bagging by additionally selecting random feature subsets at each tree split, de-correlating individual decision trees.

Boosting trains weak learners sequentially, where each successive estimator focuses on correcting the residual errors or misclassifications of preceding models. Modern gradient boosting frameworks like XGBoost, LightGBM, and CatBoost implement regularized objective functions, second-order Taylor approximations of the loss function, and histogram-based split finding for exceptional efficiency on tabular data.
