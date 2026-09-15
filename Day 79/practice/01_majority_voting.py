"""Practice 1: Hard and Soft Majority Voting from Scratch."""
import numpy as np

def hard_voting(predictions_matrix: np.ndarray) -> np.ndarray:
    """Compute hard majority voting across trees.
    predictions_matrix: shape (n_samples, n_trees)
    """
    votes_sum = np.sum(predictions_matrix, axis=1)
    threshold = predictions_matrix.shape[1] / 2.0
    return (votes_sum > threshold).astype(int)

def soft_voting(probabilities_3d: np.ndarray) -> np.ndarray:
    """Compute soft voting by averaging class probabilities across trees.
    probabilities_3d: shape (n_samples, n_trees, n_classes)
    """
    mean_probs = np.mean(probabilities_3d, axis=1)
    return np.argmax(mean_probs, axis=1)

if __name__ == '__main__':
    # 5 trees predicting on 4 samples
    preds = np.array([
        [1, 1, 0, 1, 0],  # 3 votes for 1 -> 1
        [0, 0, 0, 1, 0],  # 1 vote for 1 -> 0
        [1, 0, 1, 1, 1],  # 4 votes for 1 -> 1
        [0, 1, 0, 0, 0],  # 1 vote for 1 -> 0
    ])
    hard_res = hard_voting(preds)
    print(f"Hard voting results: {hard_res}")
    assert list(hard_res) == [1, 0, 1, 0]
    print("Practice 1 passed!")
