# 🚀 DAY 95 / 200 — ADVANCED YOLO TRAINING & EVALUATION

## 🧠 Masterclass: Precision-Recall Operating Points

### The Concept:
A detector does not just output a binary `Cat` or `Not Cat`. It outputs a continuous Confidence Score `(0.01 - 0.99)`.
If you set the Confidence Threshold very low (e.g. `0.25`), you maximize **Recall** (the detector flags everything remotely resembling a Cat), but you tank **Precision** (tons of False Positives).
If you set it very high (e.g. `0.85`), you maximize **Precision** (it's definitely a Cat), but tank **Recall** (you miss the blurry Cats). 

This is why mAP (Mean Average Precision) integrates the area under the entire Precision-Recall curve to provide a holistic metric independent of an arbitrary threshold choice.
