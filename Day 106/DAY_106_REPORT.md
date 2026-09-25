# Day 106: RNNs & Sequential Text Learning — Comprehensive Report

## 📌 1. Executive Summary

- **Mission**: Move from bag-of-words / global pooling to sequence-aware modeling using Recurrent Neural Networks (RNNs).
- **Dataset**: SMS Spam Collection (800 validated messages: 656 ham, 144 spam).
- **Partitioning**: 70% Train (560 samples), 15% Validation (120 samples), 15% Test (120 samples) with strict stratification.
- **Vocabulary**: Built strictly from training data ($V=373$ tokens, including `<PAD>=0`, `<UNK>=1`).
- **Architecture**: Trainable Embedding ($D=64$) $\to$ SimpleRNN ($H=64$, tanh) $\to$ Dropout ($p=0.3$) $\to$ Dense ($32$, ReLU) $\to$ Dropout ($p=0.2$) $\to$ Output ($1$, Sigmoid).

---

## 🛠️ 2. Environment & Framework Audit

- **Python Version**: 3.14.4 (win32)
- **PyTorch Status**: VERIFIED (PyTorch 2.14.0+cpu operational: forward, backward, RNN layers functional)
- **TensorFlow Status**: UNVERIFIED — ENVIRONMENT LIMITATION (Python 3.14.4 does not have prebuilt TensorFlow wheels on PyPI. Official TensorFlow wheels currently support up to Python 3.12.)

> [!NOTE]
> As documented in the Day 100 system audit, official prebuilt TensorFlow wheels on PyPI currently support up to Python 3.12. > On Python 3.14, deep learning execution is natively conducted using PyTorch 2.14.0+cpu while maintaining full theoretical, architectural, and NumPy fidelity.

---

## ⚖️ 3. Model Benchmark Comparison

Performance on the 120-sample held-out Test set (99 Ham, 21 Spam):

| Model | Representation | Test F1 | Accuracy | Precision | Recall | ROC-AUC | Parameters | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | Majority Class | **0.0000** | 0.8250 | 0.0000 | 0.0000 | 0.5000 | 0 | 0.001 |
| **Logistic Regression** | TF-IDF (1-2 N-grams) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1,030 | 0.011 |
| **Linear SVM** | TF-IDF (1-2 N-grams) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1,030 | 0.018 |
| **Day 105 Neural Model** | Embedding + Masked Pooling | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 28,097 | 0.452 |
| **Day 106 SimpleRNN** | Embedding + SimpleRNN (H=64) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 34,305 | 1.248 |

---

## 🔬 4. Controlled Experiments (A, B, C, D)

| Experiment | Emb Dim | Hidden Units | Seq Length | Dropout | Parameters | Train Acc | Val Acc | Test F1 | Test AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Exp_A_Compact_32` | 32 | 32 | 40 | 0.2 | 14,593 | 1.0000 | 0.9917 | **1.0000** | 1.0000 |
| `Exp_B_Baseline_64` | 64 | 64 | 40 | 0.3 | 34,305 | 1.0000 | 0.9917 | **1.0000** | 1.0000 |
| `Exp_C_HighCap_128` | 128 | 128 | 40 | 0.3 | 89,089 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| `Exp_D_SeqLen_20` | 64 | 64 | 20 | 0.3 | 34,305 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| `Exp_D_SeqLen_40` | 64 | 64 | 40 | 0.3 | 34,305 | 1.0000 | 0.9917 | **1.0000** | 1.0000 |
| `Exp_D_SeqLen_60` | 64 | 64 | 60 | 0.3 | 34,305 | 1.0000 | 0.9917 | **1.0000** | 1.0000 |
| `Exp_D_SeqLen_100` | 64 | 64 | 100 | 0.3 | 34,305 | 1.0000 | 0.9917 | **1.0000** | 1.0000 |

---

## 🎚️ 5. Decision Threshold Analysis (0.10 to 0.90)

| Threshold | Precision | Recall | F1 Score | False Positives | False Negatives |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.10` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.20` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.30` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.40` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.50` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.60` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.70` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.80` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |
| `0.90` | 1.0000 | 1.0000 | **1.0000** | 0.0 | 0.0 |

### ⚖️ Cost Trade-off Discussion:
- **False Positives (Precision cost)**: Classifying a legitimate ham message (e.g., family emergency, banking 2FA code, work email) as spam causes high user friction and potential crisis.
- **False Negatives (Recall cost)**: Letting a spam message enter the inbox causes minor annoyance, but rarely direct operational failure.
- **Recommendation**: For production spam filters, set threshold between $0.60$ and $0.70$ to prioritize high precision (minimizing false positives to zero).

---

## 🔍 6. Error Analysis

- **Total Misclassified Samples**: 0
- **Result**: Clean 100% test set separation achieved on the held-out split with threshold 0.50.

---

## 📈 7. Visualizations Generated

All 18 charts saved to `output/charts/`:

1. `1_class_distribution.png` — Ham vs Spam sample distribution.
2. `2_message_length_distribution.png` — Character length histogram.
3. `3_token_count_distribution.png` — Word token count histogram with T=40 cutoff.
4. `4_character_count_distribution.png` — Class-stratified character boxplots.
5. `5_training_loss.png` — SimpleRNN BCE loss progression across epochs.
6. `6_validation_loss.png` — Validation loss trajectory with early stopping.
7. `7_training_accuracy.png` — Training accuracy curve.
8. `8_validation_accuracy.png` — Validation accuracy convergence.
9. `9_confusion_matrix.png` — Heatmap of test set predictions.
10. `10_roc_curve.png` — Receiver Operating Characteristic curve.
11. `11_precision_recall_curve.png` — Precision-Recall curve.
12. `12_threshold_vs_f1.png` — Threshold curve vs F1 score.
13. `13_threshold_vs_precision.png` — Threshold curve vs Precision.
14. `14_threshold_vs_recall.png` — Threshold curve vs Recall.
15. `15_hidden_units_vs_f1.png` — Hidden state dimension sweep ($H=32, 64, 128$).
16. `16_sequence_length_vs_f1.png` — Max sequence length sweep ($T=20, 40, 60, 100$).
17. `17_embedding_dimension_vs_f1.png` — Embedding dimension sweep ($D=32, 64, 128$).
18. `18_model_comparison.png` — Benchmark comparison bar chart.

---
*Report auto-generated by Day 106 Neural NLP & Sequential Text Learning Engine.*