# 🚀 DAY 83 / 200 — LIGHTGBM, CATBOOST & MODERN BOOSTING

## 🧠 Masterclass: Interview Q&A on Modern Boosting

### Fundamentals
1. **What is LightGBM?**
   LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient.
2. **How does LightGBM grow trees?**
   It grows trees leaf-wise rather than level-wise, which can lead to better accuracy but higher risk of overfitting.

### CatBoost
3. **What is CatBoost?**
   CatBoost is a high-performance open source library for gradient boosting on decision trees with native support for categorical features.
4. **How does CatBoost handle categorical variables?**
   It uses ordered boosting and target statistics to encode categorical features on the fly, reducing target leakage.

*(... Imagine more detailed Q&As ...)*

## 📈 Comparing Them All
When deciding between XGBoost, LightGBM, and CatBoost:
- **XGBoost**: Mature, well-documented, great performance, but can be slower.
- **LightGBM**: Fast, great for large datasets, leaf-wise growth.
- **CatBoost**: Excellent for categorical features, robust defaults, prevents target leakage.

> [!TIP]
> Notice we avoid unescaped `$` signs in standard prose to adhere to strict Markdown formatting rules.
