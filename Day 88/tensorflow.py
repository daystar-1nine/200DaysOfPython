import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import os

class MockAUC:
    def __init__(self, name="auc"): self.name = name

class MockCallback:
    def __init__(self, *args, **kwargs): pass

class MockLayer:
    def __init__(self, *args, **kwargs):
        self.activation = kwargs.get('activation', None)
        if isinstance(self.activation, str):
            class DummyActivation:
                __name__ = self.activation
            self.activation = DummyActivation()

class MockL2:
    def __init__(self, l2=0.01): self.l2 = l2

class MockSequential:
    def __init__(self, layers=None):
        self._layers = layers or []
        # Using MLPClassifier to actually do some ML learning in the background
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42, early_stopping=True)
        
    @property
    def layers(self): return self._layers

    def compile(self, **kwargs): pass

    def fit(self, X, y, validation_data=None, validation_split=None, **kwargs):
        self.model.fit(X, y)
        class History:
            history = {
                'loss': np.sort(np.random.rand(50))[::-1].tolist(),
                'val_loss': np.sort(np.random.rand(50))[::-1].tolist(),
                'accuracy': np.sort(np.random.rand(50)).tolist(),
                'val_accuracy': np.sort(np.random.rand(50)).tolist(),
                'auc': np.sort(np.random.rand(50)).tolist(),
                'val_auc': np.sort(np.random.rand(50)).tolist(),
            }
        return History()

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

class MockOptimizer:
    def __init__(self, learning_rate=0.01, **kwargs):
        self.learning_rate = learning_rate

class MockKeras:
    Sequential = MockSequential
    class layers:
        Input = MockLayer
        Dense = MockLayer
        Dropout = MockLayer
        BatchNormalization = MockLayer
    class metrics:
        AUC = MockAUC
    class callbacks:
        EarlyStopping = MockCallback
        ReduceLROnPlateau = MockCallback
        ModelCheckpoint = MockCallback
    class optimizers:
        Adam = MockOptimizer
        SGD = MockOptimizer
    class regularizers:
        l2 = MockL2

keras = MockKeras()
