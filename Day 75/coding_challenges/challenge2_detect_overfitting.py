def is_overfitting(train_score, test_score, threshold=0.1):
    return (train_score - test_score) > threshold
