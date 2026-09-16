# Day 89: Convolutional Neural Networks (CNN)

Welcome to Computer Vision! Today we pivot from standard dense layers to discovering spatial patterns through convolution, strides, pooling, and image normalization.

## What's Inside?
- `app/`: CNN Image Classification Pipeline built to handle tensor inputs.
- `app/models/`: CNN factories comparing dense baselines against Deep Conv2D architectures incorporating Dropout, Batch Normalization, and spatial Data Augmentation!
- `output/charts/`: 15+ high-fidelity analytical charts visualizing CNN evaluation structures including a massive 10x10 Confusion Matrix over the Fashion-MNIST classes.
- `Day89.md`: Theoretical exploration of padding, kernels, and parameter-sharing mathematically!

## How to Run
```bash
cd "Day 89"
python -m app.main
pytest tests/
```
