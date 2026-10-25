# Cross-Validation and Data Leakage Prevention

K-Fold Cross-Validation splits training data into $K$ disjoint folds, training the model on $K-1$ folds and evaluating on the held-out fold iteratively.
Stratified K-Fold preserves class proportion across each partition, which is critical for imbalanced classification tasks.

Data leakage occurs when information from outside the training dataset influences model training.
To prevent leakage, all feature scalers, imputers, and dimensionality reduction projections must be fitted strictly on the training partition and only applied to validation/test folds.