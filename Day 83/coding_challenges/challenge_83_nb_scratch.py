import numpy as np

class NaiveBayesScratch:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + 1e-9 # Smoothing
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_single(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in np.array(X)])

def test_nb():
    print("Testing Naive Bayes Scratch Implementation...")
    X_train = [[1.5, 2.0], [1.1, 1.8], [6.0, 7.0], [7.1, 8.0]]
    y_train = [0, 0, 1, 1]
    
    nb = NaiveBayesScratch()
    nb.fit(X_train, y_train)
    
    X_test = [[1.2, 1.9], [6.5, 7.5]]
    preds = nb.predict(X_test)
    assert preds[0] == 0
    assert preds[1] == 1
    print("✅ Naive Bayes passed")

if __name__ == "__main__":
    test_nb()
