# 🚀 DAY 102 / 200 — NLP FEATURE ENGINEERING & CLASSICAL TEXT CLASSIFICATION

## 🎯 Overview
Day 102 advances our Natural Language Processing capabilities by building an enterprise-grade **NLP Classification Benchmark Engine**. We explore how different text feature representations (Word vs Character N-grams, varying `ngram_range`, `min_df`, `max_df`, `max_features`, and `sublinear_tf`) interact with classical machine learning algorithms:
- **Multinomial Naive Bayes**
- **Logistic Regression** (L2-regularized with varying $C$)
- **Linear Support Vector Classifier (LinearSVC)**

The benchmark strictly enforces zero data leakage via Scikit-Learn `Pipeline` integration and evaluates performance using 5-Fold Stratified Cross-Validation on held-out test splits.

---

## 📊 Status & Progress Tracker
```text
Day:              102 / 200
Completed:        102
Remaining:        98
Progress:         51% COMPLETE

█████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░
                         51%
```

---

## 🧠 Learning Objectives
- Deeply understand TF-IDF mathematics and its operational parameters (`min_df`, `max_df`, `max_features`, `sublinear_tf`).
- Compare word-level vs character-level n-gram feature spaces and understand their respective utility (semantic context vs typo/morphology robustness).
- Implement a complete **TF-IDF from scratch** with exact smoothed IDF and L2 row normalization.
- Analyze linear decision boundaries in ultra-sparse, high-dimensional spaces.
- Rigorously evaluate models with 5-Fold Stratified Cross-Validation, ROC curves, and Precision-Recall curves.
- Execute empirical error analysis on False Positives and False Negatives to uncover root causes.
- Inspect model coefficients for interpretability and discriminative keyword association.

---

## 📊 Dataset & Data Audit
The engine operates on the **SMS Spam Collection** dataset (800 curated samples with realistic class imbalance):
- **Total Records**: 800
- **Classes**: `ham` (656 samples, 82.0%), `spam` (144 samples, 18.0%)
- **Missing Values**: 0 across all columns
- **Duplicate Records**: 36 intentional duplicates preserved to evaluate deduplication resilience
- **Message Length (Characters)**: Mean = 81.3, Median = 72.0, Min = 17, Max = 159
- **Message Length (Words)**: Mean = 12.8, Median = 12.0, Min = 3, Max = 26

---

## 🧹 NLP Preprocessing & Cleaning Strategies
We empirically compared three cleaning strategies to test whether *more preprocessing = better model*:
1. **Strategy A (Minimal)**: Lowercasing and whitespace normalization only.
2. **Strategy B (Punctuation Stripped)**: Punctuation removal and URL/email stripping.
3. **Strategy C (Aggressive)**: Removal of punctuation, URLs, numeric digits, and short tokens ($< 3$ characters).

> **Empirical Finding**: Strategy B (Punctuation Stripped) yielded the optimal balance between noise reduction and preserving essential token boundaries without destroying discriminative numerical tokens (such as phone numbers and bonus amounts in spam messages).

---

## 📐 From-Scratch TF-IDF Implementation
Built in `app/features/scratch_tfidf.py` and demonstrated in `coding_challenges/challenge_102_tfidf_scratch.py`:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}} \quad \text{or} \quad \text{sublinear: } 1 + \log(f_{t,d})$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + df(t)}\right) + 1$$
$$\text{TFIDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$
$$\text{L2 Normalized: } v_{\text{norm}} = \frac{v}{\|v\|_2}$$

Tested on the canonical exercise corpus:
- $D_1$: `"python data science"`
- $D_2$: `"python machine learning"`
- $D_3$: `"machine learning python"`

**Verification**: The from-scratch output matches Scikit-Learn's `TfidfVectorizer(norm='l2', smooth_idf=True)` with zero shape or value divergence.

---

## 📊 Model Comparison Benchmark (5-Fold Stratified CV + Test Set)

| Model | Features | CV Mean (F1) | CV Std | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time (ms) | Inference Time (ms) | Vocab Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Naive Bayes** | Word TF-IDF | 0.9634 | 0.0215 | 0.9900 | 1.0000 | 0.9444 | 0.9714 | 0.9997 | 8.2 | 1.9 | 1,034 |
| **Logistic Regression** | Word TF-IDF | **0.9953** | **0.0093** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 12.4 | 2.1 | 1,034 |
| **Linear SVM** | Word TF-IDF | 0.9953 | 0.0093 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28.6 | 2.3 | 1,034 |
| **Logistic Regression** | Char TF-IDF | 0.9859 | 0.0142 | 0.9950 | 1.0000 | 0.9722 | 0.9859 | 0.9998 | 34.1 | 4.8 | 6,420 |
| **Linear SVM** | Char TF-IDF | 0.9907 | 0.0124 | 0.9950 | 1.0000 | 0.9722 | 0.9859 | 0.9998 | 72.3 | 5.1 | 6,420 |

---

## 📈 15 Analytical Visualizations Generated
Saved in [`Day 102/output/charts/`](file:///s:/Programming/Python200days/Day%20102/output/charts/):
1. `1_class_distribution.png`: Imbalance ratio between legitimate `ham` and `spam`.
2. `2_message_length_distribution.png`: Character length distributions per class.
3. `3_word_count_distribution.png`: Word count histograms per class.
4. `4_top_vocabulary_terms.png`: Top 15 most frequent corpus terms.
5. `5_top_spam_terms.png`: Top 12 positive Logistic Regression weights (`call`, `claim`, `free`, `prize`, `won`).
6. `6_top_ham_terms.png`: Top 12 negative Logistic Regression weights (`meeting`, `lunch`, `tomorrow`, `thanks`, `notes`).
7. `7_model_f1_comparison.png`: F1 score ranking across all 5 benchmark pipelines.
8. `8_model_precision_comparison.png`: Precision comparison (all models maintained 1.0000 precision!).
9. `9_model_recall_comparison.png`: Recall comparison across models.
10. `10_confusion_matrix.png`: Heatmap of best model (`Logistic Regression Word TF-IDF`).
11. `11_roc_curves.png`: Multi-model ROC curves demonstrating exceptional discriminative capacity.
12. `12_precision_recall_curves.png`: Multi-model Precision-Recall curves.
13. `13_feature_count_comparison.png`: Dimensionality comparison (Word 1,034 features vs Char 6,420 features).
14. `14_cv_mean_variation.png`: 5-fold cross-validation error bars (Mean $\pm$ Std).
15. `15_error_distribution.png`: Breakdown of False Positives vs False Negatives.

---

## 🔍 Error Analysis & Diagnostics
- **False Positives (Ham classified as Spam)**: 0 occurrences in top models.
- **False Negatives (Spam classified as Ham)**: 0 to 1 occurrences in character-level pipelines when spam messages utilized natural colloquial greetings (*"Hey, congratulations on your gift!"*).
- **Interpretability Insight**: In sparse text models, linear coefficients cleanly separate spam triggers from conversational markers.

---

## 🎤 Comprehensive Interview Questions & Answers

### NLP
1. **What is TF-IDF?**  
   Term Frequency — Inverse Document Frequency is a statistical weight reflecting how important a word is to a document in a collection. It scales up terms frequent in a single document while dampening terms that appear ubiquitously across the entire corpus.
2. **Why is TF-IDF better than raw word counts in some applications?**  
   Raw word counts heavily bias models toward generic stopwords and ubiquitous terms (like *"is"*, *"the"*, *"email"*). TF-IDF penalizes these high-document-frequency terms, highlighting informative keywords.
3. **What is document frequency?**  
   The count or fraction of documents in the corpus containing a specific term $t$: $df(t) = |\{d \in D : t \in d\}|$.
4. **What is an n-gram?**  
   A contiguous sequence of $n$ items (tokens or characters) extracted from a text string.
5. **Word n-gram vs character n-gram?**  
   - *Word n-grams* capture local syntax and multi-word phrases (e.g., *"credit card"*, *"not happy"*).
   - *Character n-grams* capture subword morphological roots, prefixes, suffixes, and are robust to spelling variations, typos, and obfuscations (e.g., *"c@sh"*, *"w1nner"*).
6. **Why are text matrices sparse?**  
   Vocabularies contain thousands to millions of unique terms, but any single document only contains a few dozen words. Over 99% of matrix cells are zeros, making compressed sparse row (CSR) representations essential.
7. **Why might stemming hurt classification?**  
   Stemming can collapse distinct semantic words into the same crude stem (e.g., *"universe"* and *"university"* might both become *"univers"*), creating ambiguity.
8. **Why shouldn't stopwords always be removed?**  
   In sentiment analysis or grammatical tasks, removing stopwords strips negations (*"not"*, *"never"*) or prepositions (*"under"*, *"over"*), which completely inverts or alters the semantic meaning.

### Machine Learning
9. **Why does Naive Bayes work well with text?**  
   Despite the unrealistic assumption that words are conditionally independent given the class, Naive Bayes only requires the correct ranking of class posterior probabilities to make the right prediction. In high-dimensional text with strong keyword cues, this ranking is reliable.
10. **Why can Linear SVM work well with high-dimensional sparse data?**  
    In high-dimensional spaces (e.g., $D > 10,000$), classes are almost always linearly separable. Linear SVM maximizes the margin between the separating hyperplane and the closest data points, providing strong generalization without incurring the computational overhead of non-linear kernels.
11. **Why does Logistic Regression work well for text?**  
    Text features are high-dimensional and sparse. Logistic Regression efficiently learns convex weights for each term, naturally handles regularization (L1/L2), and outputs well-calibrated probabilities.
12. **What does `C` mean in Logistic Regression?**  
    `C` is the inverse regularization strength ($C = 1 / \lambda$). Smaller `C` enforces stronger regularization (smaller weights, simpler model), preventing overfitting in high-dimensional vocabularies.
13. **What does `C` mean in SVM?**  
    `C` controls the trade-off between maximizing the margin and minimizing classification errors on the training set. A large `C` heavily penalizes margin violations; a smaller `C` permits a wider margin at the cost of some misclassifications.
14. **What is regularization?**  
    A mathematical penalty added to the loss function (e.g., L1 Lasso $\sum |w_i|$ or L2 Ridge $\sum w_i^2$) to constrain parameter magnitudes and prevent overfitting.
15. **Why use stratified cross-validation?**  
    In imbalanced datasets (e.g., 82% ham, 18% spam), standard cross-validation might create folds with disproportionately few (or zero) minority samples. Stratified CV ensures each fold preserves the exact class proportion of the overall dataset.

### Data Leakage
16. **Why must TF-IDF be fitted only on training data?**  
    Fitting TF-IDF computes document frequencies ($df(t)$) and vocabulary indices across the dataset. If fitted on test data, the model gains prior knowledge about word distributions in unseen evaluation data, invalidating performance estimates.
17. **Why is preprocessing inside a Pipeline useful?**  
    Encapsulating vectorization and classification inside a Scikit-Learn `Pipeline` ensures that during cross-validation, feature extraction is fitted strictly on the $k-1$ training folds and merely transformed on the validation fold.
18. **Why shouldn't the test set be used for model selection?**  
    Using the test set to choose hyperparameters or compare models leads to "information snooping," where the model indirectly overfits to the test set, producing overly optimistic estimates of real-world generalization.

### Evaluation
19. **Precision vs recall?**  
    - **Precision**: $\frac{TP}{TP + FP}$ — Of all messages predicted as spam, what fraction was actually spam?
    - **Recall**: $\frac{TP}{TP + FN}$ — Of all actual spam messages, what fraction did the model catch?
20. **Why might F1 be more informative than accuracy?**  
    In imbalanced data, a naive model predicting only the majority class can score high accuracy while offering zero utility. F1 is the harmonic mean of Precision and Recall, penalizing models that sacrifice one for the other.

---

## 💻 Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run main benchmark pipeline
python app/main.py

# Run standalone experiments
python experiments/word_vs_char.py
python experiments/preprocessing_comparison.py
python experiments/model_comparison.py

# Run coding challenges (including manual calculation verification)
python coding_challenges/challenges_1_to_10.py
python coding_challenges/challenge_102_tfidf_scratch.py

# Run 54 unit tests
pytest -v
```

---

## 🏆 Key Learnings
1. **Vocabulary Sparsity**: Character n-grams expand vocabulary size by $>6\times$ compared to word unigrams/bigrams, offering robustness to typos at the expense of higher memory and training latency.
2. **Linear Hyperplanes Excel in NLP**: High-dimensional text spaces are almost always linearly separable; Logistic Regression and Linear SVM achieve near-perfect performance with minimal inference latency ($< 3\text{ ms}$).
3. **Pipeline Integrity**: Zero data leakage via strict Scikit-Learn `Pipeline` encapsulation is the single most important architectural safeguard in NLP.
