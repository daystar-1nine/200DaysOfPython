# Regularization Techniques: L1 Lasso and L2 Ridge

Regularization prevents machine learning models from overfitting by penalizing large parameter weights in the optimization objective. L1 regularization (Lasso) adds the sum of absolute parameter weights to the loss function (lambda * sum(|w|)). Because the L1 penalty has non-differentiable sharp corners at zero, it drives non-essential coefficients exactly to zero, effectively performing embedded feature selection and producing sparse models.

L2 regularization (Ridge) adds the sum of squared parameter weights (lambda * sum(w^2)), shrinking coefficients continuously toward zero without eliminating them completely. Elastic Net combines both L1 and L2 penalties with a mixing ratio parameter, successfully handling correlated feature groups where Lasso might arbitrarily select only one predictor.
