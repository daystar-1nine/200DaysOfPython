import numpy as np

class KNNScratch:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported metric")

    def _predict_single(self, x):
        distances = [self._distance(x, x_t) for x_t in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority voting
        unique_classes, counts = np.unique(k_nearest_labels, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in np.array(X)])

def test_knn():
    print("Testing KNN Scratch Implementation...")
    X_train = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9,11]]
    y_train = [0, 0, 1, 1, 0, 1]
    
    knn = KNNScratch(k=3, metric='euclidean')
    knn.fit(X_train, y_train)
    
    X_test = [[1, 1], [8, 9]]
    preds = knn.predict(X_test)
    assert preds[0] == 0
    assert preds[1] == 1
    print("✅ KNN Euclidean passed")
    
    knn_man = KNNScratch(k=3, metric='manhattan')
    knn_man.fit(X_train, y_train)
    preds_man = knn_man.predict(X_test)
    assert preds_man[0] == 0
    assert preds_man[1] == 1
    print("✅ KNN Manhattan passed")

if __name__ == "__main__":
    test_knn()
