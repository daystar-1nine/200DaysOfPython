"""Linear Support Vector Classifier model factory."""
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def get_linear_svm_model(C: float = 1.0, random_state: int = 42, calibrate: bool = True):
    """
    Returns LinearSVC. If calibrate=True, wraps with CalibratedClassifierCV
    to provide calibrated predict_proba() estimates while retaining linear SVM decision boundary.
    """
    base_svc = LinearSVC(C=C, class_weight="balanced", random_state=random_state, max_iter=2000, dual="auto")
    if calibrate:
        return CalibratedClassifierCV(estimator=base_svc, cv=3)
    return base_svc
