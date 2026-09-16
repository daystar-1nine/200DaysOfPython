import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class MockLayer:
    def __init__(self, *args, **kwargs): pass

class MockSequential:
    def __init__(self, layers=None):
        self._layers = layers or []
        self.model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2, random_state=42)
        self.input = "mock_input"
        
    @property
    def layers(self): 
        # return dummy objects with output for feature maps
        class L:
            output = "mock_output"
        return [L() for _ in range(10)]

    def compile(self, **kwargs): pass

    def fit(self, X, y, **kwargs):
        # Flatten images for sklearn if they are 3D or 4D
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)
        class History:
            history = {
                'loss': np.random.rand(5).tolist(),
                'val_loss': np.random.rand(5).tolist(),
                'accuracy': np.random.rand(5).tolist(),
                'val_accuracy': np.random.rand(5).tolist(),
            }
        return History()

    def predict(self, X):
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        # return fake probabilities for 10 classes
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
        # Return tiny dummy dataset for rapid testing
        x_train = np.random.randint(0, 255, (100, 28, 28))
        y_train = np.random.randint(0, 10, (100,))
        x_test = np.random.randint(0, 255, (20, 28, 28))
        y_test = np.random.randint(0, 10, (20,))
        return (x_train, y_train), (x_test, y_test)

class MockModel:
    def __init__(self, inputs, outputs): pass
    def predict(self, X):
        return [np.random.rand(1, 28, 28, 32) for _ in range(3)]

class MockKeras:
    Sequential = MockSequential
    Model = MockModel
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
    class datasets:
        fashion_mnist = MockFashionMNIST()
    class callbacks:
        EarlyStopping = MockLayer

keras = MockKeras()
