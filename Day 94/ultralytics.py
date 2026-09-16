
import numpy as np

class MockBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy)]

class MockResult:
    def __init__(self):
        self.boxes = [
            MockBox(0, 0.94, [100, 50, 400, 350]),
            MockBox(1, 0.88, [200, 150, 300, 450])
        ]
    def plot(self):
        return np.zeros((640, 640, 3), dtype=np.uint8)

class YOLO:
    def __init__(self, model_name):
        self.model_name = model_name
        
    def __call__(self, source):
        # mock inference
        return [MockResult()]
        
    def train(self, data, epochs, imgsz, batch):
        # mock training
        pass
