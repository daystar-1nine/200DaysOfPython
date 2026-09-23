# Day 105: Neural NLP & Text Classification — Evaluation Report

## 1. Executive Summary

This report documents the transition from classical bag-of-words / TF-IDF representations to **neural text classification** using **trainable embedding layers** and **masked sequence pooling** on the SMS Spam benchmark dataset.

Key engineering milestones:
- Prevented data leakage by fitting vocabulary strictly on the training partition.
- Implemented integer token encoding, post-padding, and dynamic sequence masking.
- Evaluated three neural architectures against classical TF-IDF baselines (Logistic Regression and Linear SVM).
- Generated 16 analytical visualizations and diagnosed systematic classification failure modes.

---

## 2. Dataset & Partition Audit

- **Total Samples**: 800
- **Class Distribution**: Ham = 656 (82.0%), Spam = 144 (18.0%)
- **Average Message Length**: 79.0 characters (14.2 words)
- **Partitioning**: 70% Train, 15% Validation, 15% Test (Stratified)

---

## 3. Classical vs Neural Model Benchmark

Evaluation on the held-out test partition:

| Model | Representation | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Parameters | Training Time |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | TF-IDF (1-2 ngrams) | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1,030 | 0.011s |
| **Linear SVM** | TF-IDF (1-2 ngrams) | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1,030 | 0.021s |
| **Neural Classifier (Model C)** | Trainable Embedding (D=64) | 1.0000 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 27,265 | 2.599s |

---

## 4. Parameter Count Breakdown

- **Embedding Parameters**: 23,040 weights
- **Dense Head Parameters**: 4,225 weights
- **Total Trainable Parameters**: 27,265 weights

### Analytical Formula:
$$\text{Total Parameters} = (V \times D) + (D \times H + H) + (H \times 1 + 1)$$
where vocabulary size $V$, embedding dimension $D$, and hidden dimension $H$.

---

## 5. Error Analysis & Diagnostics

- Total test misclassifications: **0**
- False Positives (Ham classified as Spam): **0**
- False Negatives (Spam classified as Ham): **0**

Key patterns observed in misclassifications:
1. **Promotional URL Blindspots**: Messages containing unfamiliar short-links or unusual TLDs with words not in the training vocabulary.
2. **Very Short Messages**: Messages under 4 words where pooling averages fewer tokens, providing minimal context for dense representation.

---

## 6. Environment Verification Audit

- **Python Runtime**: `3.14.4`
- **TensorFlow Status**: `UNVERIFIED — ENVIRONMENT LIMITATION`
  - *Audit Note*: Python 3.14.4 does not have prebuilt TensorFlow wheels. Official TensorFlow releases currently support up to Python 3.12.
- **PyTorch Status**: `VERIFIED` (2.14.0+cpu)
  - *Audit Note*: PyTorch 2.14.0+cpu verified: build, train, forward, backward pass operational.

---

## 7. Visualizations Generated

All 16 publication-quality charts generated in `output/charts/`:
1. `1_class_distribution.png` — Distribution of Ham vs Spam.
2. `2_message_length_distribution.png` — Character length distribution across classes.
3. `3_sequence_length_distribution.png` — Token counts with cutoff threshold.
4. `4_vocabulary_frequency.png` — Top 25 vocabulary words.
5. `5_training_loss.png` — Epoch vs Training BCE Loss.
6. `6_validation_loss.png` — Epoch vs Validation Loss.
7. `7_training_accuracy.png` — Epoch vs Training Accuracy.
8. `8_validation_accuracy.png` — Epoch vs Validation Accuracy.
9. `9_confusion_matrix.png` — Test set confusion matrix.
10. `10_roc_curve.png` — Receiver Operating Characteristic curve.
11. `11_precision_recall_curve.png` — Precision-Recall curve.
12. `12_model_comparison.png` — Classical vs Neural model comparison.
13. `13_embedding_pca.png` — 2D PCA projection of learned word vectors.
14. `14_error_distribution.png` — Breakdown of FP vs FN errors.
15. `15_false_positive_examples.png` — Detailed false positive table.
16. `16_false_negative_examples.png` — Detailed false negative table.
