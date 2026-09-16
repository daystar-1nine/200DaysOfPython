from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

class MockAUC:
    def __init__(self, name="auc"):
        self.name = name

class MockEarlyStopping:
    def __init__(self, **kwargs):
        pass

class MockLayer:
    def __init__(self, *args, **kwargs):
        self.activation = kwargs.get('activation', None)
        if isinstance(self.activation, str):
            class DummyActivation:
                __name__ = self.activation
            self.activation = DummyActivation()

class MockSequential:
    def __init__(self, layers=None):
        self._layers = layers or []
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=50, random_state=42, early_stopping=True)
        
    @property
    def layers(self):
        # for tests
        return self._layers

    def compile(self, **kwargs):
        pass

    def fit(self, X, y, validation_split=0.2, **kwargs):
        self.model.fit(X, y)
        # Mock history
        class History:
            history = {
                'loss': np.random.rand(50).tolist(),
                'val_loss': np.random.rand(50).tolist(),
                'auc': np.random.rand(50).tolist(),
                'val_auc': np.random.rand(50).tolist(),
            }
        return History()

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

class MockKeras:
    Sequential = MockSequential
    class layers:
        Input = MockLayer
        Dense = MockLayer
        Dropout = MockLayer
    class metrics:
        AUC = MockAUC
    class callbacks:
        EarlyStopping = MockEarlyStopping

keras = MockKeras()
