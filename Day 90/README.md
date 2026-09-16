# Day 90: Advanced CNN Training Engine

Welcome to the 90-day milestone! Today we transitioned from a standard CNN to a production-grade CV Training Pipeline.

## What's Inside?
- `app/`: Fully engineered CNN ecosystem implementing Data Augmentation, Regularization (L2, Dropout), and internal Batch Normalization.
- `app/training/`: Training controllers using callbacks like `ReduceLROnPlateau`, `EarlyStopping`, and `ModelCheckpoint`.
- `output/experiments.csv`: Tracked experimental logs across 7 different architectural tests.
- `output/misclassified_predictions.csv`: Error analysis focusing specifically on structural confusions between classes alongside confidence metrics.
- `output/charts/`: 20+ analytical visualizations spanning class distributions, evaluation metrics, and feature maps.

## How to Run
```bash
cd "Day 90"
python -m app.main
pytest tests/
```
