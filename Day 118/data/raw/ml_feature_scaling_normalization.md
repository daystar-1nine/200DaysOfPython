# Feature Scaling: Standardization vs Min-Max Normalization

Distance-based algorithms (such as KNN, SVMs, and K-Means) and gradient-based models are sensitive to differences in feature magnitudes.
Standardization (Z-score normalization) transforms features to have zero mean and unit variance: $z = (x - \mu) / \sigma$.
Min-Max normalization scales values into a fixed bounded interval $[0, 1]$: $x_{norm} = (x - x_{min}) / (x_{max} - x_{min})$.

Standardization is preferred when features exhibit Gaussian distributions or contain outliers, as Min-Max scaling compresses inliers into a tiny range when extreme outliers are present.