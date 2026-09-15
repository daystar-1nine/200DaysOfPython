"""Challenge 6: Complete End-to-End Comparison of Ensemble Architectures."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score

def compare_architectures():
    X, y = make_classification(n_samples=1500, n_features=15, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
        'Bagging Classifier': BaggingClassifier(n_estimators=50, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=50, max_depth=6, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        results[name] = f1
        print(f"{name:22s} -> F1 Score: {f1:.4f}")
        
    return results

if __name__ == '__main__':
    res = compare_architectures()
    assert res['Random Forest'] > res['Decision Tree']
    print("Challenge 6 passed!")
