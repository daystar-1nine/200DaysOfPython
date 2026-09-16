
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor, clf):
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

def get_models(preprocessor):
    return {
        'Logistic Baseline': build_pipeline(preprocessor, LogisticRegression(max_iter=1000)),
        'KNN Baseline': build_pipeline(preprocessor, KNeighborsClassifier()),
        'Linear SVM': build_pipeline(preprocessor, SVC(kernel="linear", probability=True, random_state=42)),
        'RBF SVM': build_pipeline(preprocessor, SVC(kernel="rbf", probability=True, random_state=42)),
        'Polynomial SVM': build_pipeline(preprocessor, SVC(kernel="poly", degree=3, probability=True, random_state=42))
    }
