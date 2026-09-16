import numpy as np
from sklearn.neural_network import MLPClassifier

class MockLayer:
    def __init__(self, *args, **kwargs): pass

class MockCallback:
    def __init__(self, *args, **kwargs): pass

class MockHistory:
    def __init__(self):
        self.history = {
            'loss': np.random.rand(5).tolist(),
            'val_loss': np.random.rand(5).tolist(),
            'accuracy': np.random.rand(5).tolist(),
            'val_accuracy': np.random.rand(5).tolist(),
        }

class MockSequential:
    def __init__(self, layers=None):
        self._layers = layers or []
        self.model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2, random_state=42)
        
    @property
    def layers(self):
        class L:
            output = "mock_output"
        return [L() for _ in range(10)]
        
    def compile(self, **kwargs): pass

    def fit(self, X, y, **kwargs):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)
        return MockHistory()

    def predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        preds = self.model.predict_proba(X)
        if preds.shape[1] < 10:
            padded = np.zeros((preds.shape[0], 10))
            padded[:, :preds.shape[1]] = preds
            return padded
        return preds
        
    def evaluate(self, X, y, **kwargs):
        return [0.1, 0.9] # loss, accuracy

class MockFashionMNIST:
    def load_data(self):
        x_train = np.random.randint(0, 255, (200, 28, 28))
        y_train = np.random.randint(0, 10, (200,))
        x_test = np.random.randint(0, 255, (50, 28, 28))
        y_test = np.random.randint(0, 10, (50,))
        return (x_train, y_train), (x_test, y_test)

def mock_l2(*args, **kwargs):
    return "l2"

class MockKeras:
    Sequential = MockSequential
    class layers:
        Input = MockLayer
        Dense = MockLayer
        Dropout = MockLayer
        BatchNormalization = MockLayer
        Conv2D = MockLayer
        MaxPooling2D = MockLayer
        Flatten = MockLayer
        RandomFlip = MockLayer
        RandomRotation = MockLayer
        RandomZoom = MockLayer
        RandomTranslation = MockLayer
        ReLU = MockLayer
    class datasets:
        fashion_mnist = MockFashionMNIST()
    class callbacks:
        EarlyStopping = MockCallback
        ReduceLROnPlateau = MockCallback
        ModelCheckpoint = MockCallback
    class optimizers:
        Adam = MockCallback
    class regularizers:
        l2 = mock_l2

keras = MockKeras()
