import numpy as np
from sklearn.neural_network import MLPClassifier

class MockLayer:
    def __init__(self, *args, **kwargs): 
        self.trainable = True
    def __call__(self, inputs, *args, **kwargs):
        return inputs

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

class MockModel:
    def __init__(self, inputs=None, outputs=None, layers=None):
        self._layers = layers or []
        self.inputs = inputs
        self.outputs = outputs
        self.model = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2, random_state=42)
        
    def __call__(self, inputs, *args, **kwargs):
        return inputs
        
    @property
    def layers(self):
        class L:
            def __init__(self):
                self.trainable = True
        return [L() for _ in range(50)]
        
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

class MockSequential(MockModel):
    pass

class MockFashionMNIST:
    def load_data(self):
        x_train = np.random.randint(0, 255, (200, 28, 28))
        y_train = np.random.randint(0, 10, (200,))
        x_test = np.random.randint(0, 255, (50, 28, 28))
        y_test = np.random.randint(0, 10, (50,))
        return (x_train, y_train), (x_test, y_test)

def mock_l2(*args, **kwargs):
    return "l2"
    
class MockTensor:
    def __init__(self, arr):
        self.arr = arr
    def numpy(self):
        return self.arr
    @property
    def shape(self):
        return self.arr.shape

def mock_resize(images, size, *args, **kwargs):
    if isinstance(images, MockTensor): images = images.arr
    N = images.shape[0]
    C = images.shape[-1]
    return MockTensor(np.random.rand(N, size[0], size[1], C))

def mock_grayscale_to_rgb(images, *args, **kwargs):
    if isinstance(images, MockTensor): images = images.arr
    N, H, W, C = images.shape
    return MockTensor(np.random.rand(N, H, W, 3))

class MockMobileNetV2(MockModel):
    def __init__(self, include_top=False, weights="imagenet", input_shape=(224, 224, 3)):
        super().__init__()
        self.trainable = True
        
    def __call__(self, inputs, *args, **kwargs):
        return inputs

class MockKeras:
    Sequential = MockSequential
    Model = MockModel
    Input = MockLayer
    class applications:
        MobileNetV2 = MockMobileNetV2
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
        GlobalAveragePooling2D = MockLayer
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

class MockImage:
    @staticmethod
    def resize(*args, **kwargs):
        return mock_resize(*args, **kwargs)
        
    @staticmethod
    def grayscale_to_rgb(*args, **kwargs):
        return mock_grayscale_to_rgb(*args, **kwargs)

def convert_to_tensor(x, *args, **kwargs):
    return MockTensor(x)

keras = MockKeras()
image = MockImage()
