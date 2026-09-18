# Knowledge Gaps — Days 1–100

### Backend / SQLAlchemy
**Topic**: Database Models and Pydantic Validation
**Evidence**: Day 50 `app/models/task.py` and `app/schemas/task.py` are left as `TODO` / `pass`.
**Why revision is needed**: Building a FastAPI backend relies entirely on the successful bridge between SQLAlchemy ORM models and Pydantic schemas. Avoiding this prevents data persistence.
**Recommended revision exercise**: Write a simple SQLite CRUD app with FastAPI, explicitly defining standard models and response models.

### Machine Learning
**Topic**: Algorithm Mathematics (From Scratch)
**Evidence**: Day 82-87 `coding_challenges` are entirely blank.
**Why revision is needed**: Calling `from sklearn.svm import SVC` is easy, but understanding *why* an RBF kernel scales dimensions requires implementing the math. The fact that the "from scratch" files are blank implies a reliance on APIs over mathematical intuition.
**Recommended revision exercise**: Implement K-Nearest Neighbors using purely NumPy arrays and `math.dist` without importing `sklearn`.

### Deep Learning
**Topic**: Forward Propagation and Loss Functions
**Evidence**: Day 87 `experiments/forward_propagation.py` and `loss_from_scratch.py` are empty.
**Why revision is needed**: Deep Learning models are black boxes if you don't understand how `Weights * Inputs + Bias` cascades through an activation function, or how Categorical Cross-Entropy penalizes wrong predictions.
**Recommended revision exercise**: Build a 1-hidden-layer MLP using only NumPy to classify a dummy dataset.
