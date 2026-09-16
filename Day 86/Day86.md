# 🚀 DAY 86 / 200 — SUPPORT VECTOR MACHINES (SVM)

## 🧠 Masterclass: Interview Q&A on SVM

### Fundamentals
1. **What is SVM?**
   Support Vector Machine is a supervised algorithm that finds a hyperplane optimizing the margin between different classes.
2. **What are Support Vectors?**
   They are the data points lying closest to the decision boundary. They dictate the margin. If you remove all other points, the boundary doesn't change.

### Kernels & Hyperparameters
3. **What is the Kernel Trick?**
   It computes the dot product of two vectors in a higher-dimensional space without explicitly transforming them into that space, making non-linear separation computationally feasible.
4. **What does 'C' control?**
   The 'C' parameter controls regularization. High C = strict margin (fewer misclassifications but smaller margin). Low C = soft margin (allows misclassifications for a wider margin).
5. **What does 'Gamma' control?**
   In RBF kernels, Gamma defines how far the influence of a single training example reaches. Low values mean 'far' (broad boundary), high values mean 'close' (tight, wiggly boundary).

> [!TIP]
> Notice we avoid unescaped `$` signs in standard prose to adhere to strict Markdown formatting rules.
