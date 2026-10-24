# Pandas Split-Apply-Combine and GroupBy Operations

The Split-Apply-Combine paradigm divides tabular datasets into discrete groups based on specified grouping keys, applies transformation or reduction functions to each group independently, and combines the results into a unified structure. The .groupby() method supports aggregate functions (mean, sum, count, std), transformations that retain the original DataFrame index, and filtering operations that discard entire groups failing specific predicate criteria.

Rolling window calculations allow computing moving averages, rolling standard deviations, and cumulative sums across ordered temporal data. Combining groupby with rolling or expanding windows enables powerful localized statistical feature engineering for time series forecasting.
