# Handling Missing Data and Outliers in Tabular Datasets

Real-world datasets frequently contain missing values represented as NaN or None. Pandas provides .isna(), .fillna(), and .dropna() for missing value imputation and deletion. Imputation strategies range from simple mean, median, or mode replacement to forward/backward filling for time series, and iterative algorithms like MICE (Multivariate Imputation by Chained Equations).

Outliers are typically detected using parametric Z-scores (values beyond 3 standard deviations from the mean) or non-parametric Interquartile Range (IQR) fences (values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR). Capping outliers via Winsorization or transforming skewed distributions via log or Box-Cox transformations prevents extreme values from distorting linear models.
