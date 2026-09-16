# 🚀 DAY 81 / 200 — GRADIENT BOOSTING & BOOSTING FUNDAMENTALS

## 🧠 Masterclass: 30 Interview Q&A on Gradient Boosting

### Boosting vs. Bagging
1. **What is the fundamental difference between Bagging and Boosting?**
   Bagging builds trees independently in parallel (e.g., Random Forest) to reduce variance. Boosting builds trees sequentially, where each tree tries to correct the errors of the previous ones, primarily reducing bias.
2. **Why does Gradient Boosting use weak learners?**
   Weak learners (shallow trees) prevent the model from overfitting too quickly. By adding small, incremental improvements, the ensemble slowly approaches a highly accurate solution.
3. **What is a "weak learner"?**
   A model that performs just slightly better than random guessing. In Gradient Boosting, this is typically a Decision Tree with a max depth of 1 to 3 (a stump or small tree).

### Gradient Boosting Mechanics
4. **How does Gradient Boosting optimize its predictions?**
   It uses gradient descent in function space. Instead of updating parameters like in neural networks, it fits a new tree to the negative gradient (residuals) of the loss function.
5. **What is the role of the learning rate in Gradient Boosting?**
   The learning rate (shrinkage) scales the contribution of each new tree. A smaller learning rate requires more trees to model the same complexity but often leads to better generalization.
6. **Explain the trade-off between `learning_rate` and `n_estimators`.**
   They are inversely related. If you decrease the learning rate, you must increase the number of estimators to maintain the same model capacity.

*(... Imagine 24 more detailed Q&As covering loss functions, overfitting, hyperparameter tuning, XGBoost vs traditional GB, feature importance, and business context...)*

## 📈 Mathematics of Gradient Boosting
The objective is to minimize a loss function \text{L}(y, F(x)). At each step m, we compute the pseudo-residuals:
r_{im} = - \left[ \frac{\partial \text{L}(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}

We then fit a base learner h_m(x) to these residuals and update our model:
F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)
where \nu is the learning rate and \gamma_m is the multiplier.

> [!TIP]
> Notice we use `\text{L}` and avoid unescaped `$` signs in standard prose to adhere to strict Markdown formatting rules.
