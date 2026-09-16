# Day 88: Practical Neural Network Training

Today we pivot from understanding Neural Networks to effectively **Training** them via systematic Optimization, Regularization, and Hyperparameter tuning!

## What's Inside?
- `app/`: Neural Network Training Pipeline Engine.
- `experiments/`: Tracked optimizations over SGD vs Adam, Learning Rates, and Batch Size.
- `app/training/trainer.py`: Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
- `app/regularization/`: Integrated Dropout and L2 kernels in dense graphs.
- `output/experiment_results.csv`: Documented historical logs across training configs.
- `output/charts/`: 20+ analytical charts visualizing optimization trajectories and generalization gaps.

## How to Run
```bash
cd "Day 88"
python -m app.main
pytest tests/
```
