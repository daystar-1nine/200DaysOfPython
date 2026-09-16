
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def get_baseline_models():
    return {
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42, max_iter=1000))]),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
def get_knn_models():
    return {
        'KNN (Unscaled)': KNeighborsClassifier(n_neighbors=5),
        'KNN (Scaled)': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))]),
        'KNN (Weighted)': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=15, weights='distance'))]),
        'KNN (Manhattan)': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=15, metric='manhattan', weights='distance'))])
    }
